from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

from src.core.documents import DocumentElement, ParsedDocument, build_snapshot
from src.core.uploads import UploadRecord
from src.infra.tools.local_rag import build_upload_retriever, build_upload_search_tool


class _LengthEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.0]


@pytest.fixture
def uploaded(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: _LengthEmbeddings())

    @contextmanager
    def create(source, *, session="requirements", notebook=False):
        path = tmp_path / "uploads" / session / ("source.ipynb" if notebook else "source.py")
        path.parent.mkdir(parents=True, exist_ok=True)
        if notebook:
            payload = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
                {"id": f"source-{index}", "cell_type": "code", "source": text,
                 "metadata": {}, "execution_count": None, "outputs": []}
                for index, text in enumerate(source)
            ]}
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_bytes(source.encode("utf-8"))
        content = path.read_bytes()
        file_id = uuid4().hex
        record = UploadRecord(
            file_id=file_id, name=path.name, path=str(path), size_bytes=len(content),
            content_hash="sha256:" + hashlib.sha256(content).hexdigest(),
            source_uri=f"upload:///{session}/{file_id}",
        )
        handle = build_upload_retriever([record], session_id=session, generation=uuid4().hex, api_key="test-key")
        try:
            yield handle.retriever
        finally:
            handle.cleanup()

    return create


def _requirement(symbols, match="definition"):
    from src.core.planner_schema import RetrievalRequirement
    return RetrievalRequirement(symbols=symbols, match=match)


def _source_retriever(source):
    document = ParsedDocument(
        snapshot=build_snapshot(source_uri="upload:///fragments/source.py", title="source.py",
                                media_type="text/x-python", source_type="upload", content=source,
                                parser="python", parser_version="test"),
        elements=[DocumentElement(element_id="source", kind="code", text=source)],
    )
    return SimpleNamespace(source_documents=(document,))


def _search_code_aspect(source, aspect, *, match="definition"):
    requirement = _requirement(["target"], match)
    requirement.aspects = [aspect]
    return build_upload_search_tool()(query=f"target {aspect}", requirement=requirement,
                                      retriever=_source_retriever(source))


@pytest.mark.parametrize(("aspect", "source"), [
    ("value=0.10", "def target(value=0.10):\n    return value\n"),
    ("'is None'", "def target():\n    return 'is None'\n"),
    ("is None", "def target(value):\n    return value is  None\n"),
    ("is None", "def target(value):\n    return (value is\n            None)\n"),
    ("value\nother", "def target(value, other):\n    value\n    other\n"),
    ("helper(", "def target():\n    return helper()\n"),
])
def test_exact_code_fragment_is_covered_within_selected_definition(aspect, source):
    payload = _search_code_aspect(source, aspect)

    assert payload.diagnostics.answerability == "covered"
    assert payload.diagnostics.missing_requirements == []
    evidence, = [hit.evidence for hit in payload.hits]
    assert evidence.excerpt == source.rstrip("\n")
    assert evidence.excerpt == source[evidence.selection.start:evidence.selection.end]


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_code_fragment_coverage_preserves_unicode_and_original_line_endings(newline):
    definition = newline.join(["def target(값):", "    설명 = '한글'; return 값 is None"])
    source = "소개 = '원문'" + newline + definition + newline
    payload = _search_code_aspect(source, "is None")

    assert payload.diagnostics.answerability == "covered"
    evidence, = [hit.evidence for hit in payload.hits]
    assert evidence.excerpt == definition
    assert evidence.excerpt == source[evidence.selection.start:evidence.selection.end]
    assert evidence.element.text == source


@pytest.mark.parametrize(("aspect", "source", "match"), [
    ("is None", "def target(value):\n    # value is None\n    return value\n", "definition"),
    ("is None", "def target():\n    return 'is None'\n", "definition"),
    ("is None", 'def target():\n    return f"is None"\n', "definition"),
    ("is None", "def target(value):\n    return value is NoneType\n", "definition"),
    ("is None", "def target(value):\n    return value\n\ndef other(value):\n    return value is None\n", "definition"),
    ("is None", "other = value is None; target(value)\n", "symbol"),
    ("discount", "def target():\n    # discount\n    return 'discount'\n", "definition"),
    ("return value", "def target(value):\n    return\n    value\n", "definition"),
    ("is None", "def target(value):\n    return value is \\\n        None\n", "definition"),
    ("value=", "def target(value=0.10):\n    # value=\n    return value\n", "definition"),
    ("helper('unfinished", "def target():\n    return helper()\n", "definition"),
    ('helper("""unfinished', "def target():\n    return helper()\n", "definition"),
])
def test_code_aspect_requires_whole_code_tokens_in_the_selected_occurrence(aspect, source, match):
    payload = _search_code_aspect(source, aspect, match=match)

    assert payload.hits
    assert payload.diagnostics.answerability == "unknown"
    assert "source_requirement_constraints_unverified" in payload.diagnostics.warnings


@pytest.mark.parametrize(("aspect", "signature", "covered"), [
    ("value=", "value=0.10", False),
    ("=None", "value=None", False),
    ("value=", "value= 0.10", True),
    ("=None", "value =None", True),
])
def test_fragment_coverage_keeps_the_original_literal_boundaries(aspect, signature, covered):
    from src.core.planner_schema import RetrievalTask
    from src.runtime.nodes.synthesis.evidence_selection import requirement_coverage

    source = f"def target({signature}):\n    return value\n"
    payload = _search_code_aspect(source, aspect)
    task = RetrievalTask(route="upload", query=f"target {aspect}", k=4,
                         requirement={"symbols": ["target"], "match": "definition", "aspects": [aspect]})
    evidence = [hit.evidence for hit in payload.hits]
    coverage = requirement_coverage(task, evidence, {item.id: [task.requirement_id] for item in evidence})

    assert payload.diagnostics.answerability == ("covered" if covered else "unknown")
    assert coverage["is_partial"] is not covered
    assert coverage["missing_aspects"] == ([] if covered else [aspect])


def test_planned_code_fragment_reaches_source_retrieval_without_losing_the_constraint():
    from src.core.contracts.boundary.graph import build_graph_state_input
    from src.runtime.nodes.planner import make_planner_node
    from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
    from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
    from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs
    from src.runtime.nodes.synthesis.evidence_selection import requirement_coverage
    from tests.core.helpers import _CapturePlannerLLM

    source = "def target(value):\n    return value is None\n"
    query = "첨부한 target 함수의 `is None` 코드 조각을 찾아 인용해줘."
    planner = _CapturePlannerLLM({"use_retrieval": True, "tasks": [{
        "route": "upload", "query": query, "k": 4,
        "requirement": {"symbols": ["target"], "match": "definition", "aspects": ["is None"]},
    }]})
    state = build_graph_state_input(user_input=query, retriever=_source_retriever(source))

    state.update(make_planner_node(planner, False)(state))
    state.update(make_retrieve_dispatch_node(None, build_upload_search_tool(), False)(state))

    task, = state["planner"].output.tasks
    assert task.requirement.aspects == ["is None"]
    diagnostic, = state["debug"].retrieval_diagnostics
    assert diagnostic.answerability == "covered"
    assert [hit.evidence.excerpt for hit in state["retrieval"].hit_log] == [source.rstrip("\n")]
    prepared = prepare_synthesis_inputs(state=state, context=build_synthesis_context(state=state),
                                        budget=RetrievedEvidenceBudget(max_excerpt_chars=480,
                                            max_total_excerpt_chars=3000, max_items=8), max_turns=6)
    assert prepared.missing_requirement_ids == []
    assert [item.excerpt for item in prepared.evidence_packet] == [source.rstrip("\n")]
    assert requirement_coverage(task, prepared.evidence_packet, prepared.evidence_requirement_map)["is_partial"] is False


def test_explicit_definition_request_does_not_quote_a_comment_or_a_call(uploaded):
    """A requested function definition is missing even when its name occurs in a comment and call."""
    source = "# archive_orders is supplied by another package\narchive_orders(items)\n\ndef archive_users():\n    return []\n"
    with uploaded(source) as retriever:
        payload = build_upload_search_tool()(requirement=_requirement(["archive_orders"]), query="archive_orders 함수 정의를 원문 그대로 발췌해줘", retriever=retriever)
    assert payload.hits == []
    assert payload.diagnostics.answerability == "missing"
    assert payload.diagnostics.missing_requirements == ["archive_orders"]


def test_complete_method_definition_is_found_outside_vector_top_k_with_exact_offsets(uploaded):
    """Exact source lookup retrieves a complete decorated method regardless of vector candidate limits."""
    method = "    @staticmethod\r\n    async def summarize_rows(rows):\r\n        label = '한글'\r\n" + "        rows = list(rows)\r\n" * 35 + "        return rows\r\n"
    source = "unrelated = 1\r\n" + "# other setup\r\n" * 90 + "class ReportBuilder:\r\n" + method + "\r\ndef other():\r\n    return None\r\n"
    with uploaded(source) as retriever:
        candidates = retriever.vectorstore.similarity_search_with_score("메서드를 발췌", k=1)
        assert all("summarize_rows" not in doc.excerpt for doc, _score in candidates)
        payload = build_upload_search_tool()(query="메서드를 발췌", k=1, retriever=retriever,
                                            requirement=_requirement(["ReportBuilder.summarize_rows"]))
    hits = payload.hits
    assert payload.diagnostics.answerability == "covered"
    assert [hit.evidence.excerpt for hit in hits] == [method.rstrip("\r\n")]
    assert hits[0].evidence.excerpt == source[hits[0].evidence.selection.start:hits[0].evidence.selection.end]
    assert hits[0].evidence.element.text == source


def test_notebook_parameter_usage_preserves_exact_source_and_cell_location(uploaded):
    """A call spanning index windows remains a complete, correctly located usage quotation."""
    call = "model = train_model(\n" + "    data,\n" * 100 + "    learning_rate=0.01,\n    epochs=20,\n)"
    setup = "# setup\n" * 20
    source = setup + call + "\n"
    requirement = _requirement(["train_model"], "symbol")
    requirement.aspects = ["learning_rate", "epochs"]
    contents = ["unrelated = 1\n", source]

    with uploaded(contents, notebook=True) as retriever:
        payload = build_upload_search_tool()(query="train_model parameters", k=1,
                                            requirement=requirement, retriever=retriever)

    assert payload.diagnostics.answerability == "covered"
    assert payload.diagnostics.missing_requirements == []
    assert [hit.evidence.excerpt for hit in payload.hits] == [call]
    evidence = payload.hits[0].evidence
    assert evidence.element.text == source
    assert evidence.selection.start == len(setup)
    assert evidence.selection.end == len(setup) + len(call)
    assert evidence.element.anchors[0].cell_index == 1
    assert evidence.element.anchors[0].cell_id == "source-1"


def test_source_lookup_remains_isolated_between_uploaded_sessions(uploaded):
    """Exact source lookup uses the current retriever and never another session's definitions."""
    with uploaded("def private_aggregate():\n    return 42\n", session="alpha") as first:
        with uploaded("def public_aggregate():\n    return 7\n", session="beta") as second:
            found = build_upload_search_tool()(query="private_aggregate", retriever=first,
                                              requirement=_requirement(["private_aggregate"]))
            absent = build_upload_search_tool()(query="private_aggregate", retriever=second,
                                               requirement=_requirement(["private_aggregate"]))
    assert found.diagnostics.answerability == "covered"
    assert absent.hits == []
    assert absent.diagnostics.answerability == "missing"


def test_planned_symbol_search_preserves_implementation_and_its_actual_example(uploaded):
    """An example call must not displace the default and calculation needed to explain its result."""
    from src.core.contracts.boundary.graph import build_graph_state_input
    from src.core.planner_schema import PlannerOutput, RetrievalTask
    from src.runtime.nodes.planner import make_planner_node
    from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
    from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
    from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs
    from tests.core.helpers import _CapturePlannerLLM

    definition = "def calculate_total(prices, discount=0.10):\n    return round(sum(prices) * (1 - discount), 2)"
    example = "example = calculate_total([10000, 20000])"
    source = definition + "\n\n" + example + "\n"
    query = "첨부한 calculate_total의 계산 방식, 기본 할인율과 실제 예제 결과를 알려줘."
    task = RetrievalTask(route="upload", query=query, k=4,
                         requirement={"symbols": ["calculate_total"], "match": "symbol"})
    with uploaded(source) as retriever:
        state = build_graph_state_input(user_input=query, retriever=retriever)
        state.update(make_planner_node(_CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[task])), False)(state))
        state.update(make_retrieve_dispatch_node(None, build_upload_search_tool(), False)(state))
    hits = state["retrieval"].hit_log
    assert {hit.evidence.excerpt for hit in hits} == {definition, example}
    assert all(hit.evidence.excerpt == source[hit.evidence.selection.start:hit.evidence.selection.end] for hit in hits)
    diagnostic, = state["debug"].retrieval_diagnostics
    assert diagnostic.answerability == "covered"
    assert diagnostic.final_evidence_count == 2
    budget = RetrievedEvidenceBudget(max_excerpt_chars=480, max_total_excerpt_chars=3000, max_items=8)
    prepared = prepare_synthesis_inputs(state=state, context=build_synthesis_context(state=state),
                                        budget=budget, max_turns=6)
    assert {item.excerpt for item in prepared.evidence_packet} == {definition, example}
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= budget.max_total_excerpt_chars
