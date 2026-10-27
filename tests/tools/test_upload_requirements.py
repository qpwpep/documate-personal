from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

from src.core.contracts import GraphState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.retrieval import UploadRetriever
from src.core.documents import DocumentElement, ParsedDocument, build_snapshot
from src.core.planner_schema import PlannerOutput
from src.core.uploads import UploadRecord
from src.infra.tools.local_rag import build_upload_retriever, build_upload_search_tool
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.synthesis.evidence_selection import requirement_coverage
from tests.core.helpers import _CapturePlannerLLM
from tests.synthesis_fixtures import prepare_synthesis_state, synthesis_excerpt_limits


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
    ("helper('unfinished", "def target():\n    return helper()\n", "definition"),
    ('helper("""unfinished', "def target():\n    return helper()\n", "definition"),
])
def test_code_aspect_requires_whole_code_tokens_in_the_selected_occurrence(aspect, source, match):
    payload = _search_code_aspect(source, aspect, match=match)

    assert payload.hits
    assert payload.diagnostics.answerability == "unknown"
    assert "source_requirement_constraints_unverified" in payload.diagnostics.warnings


@pytest.mark.parametrize(("aspect", "signature"), [
    ("value=", "value=0.10"),
    ("=None", "value=None"),
    ("value=", "value= 0.10"),
    ("=None", "value =None"),
])
def test_fragment_coverage_uses_identifier_boundaries_only_at_identifier_edges(aspect, signature):
    from src.core.planner_schema import RetrievalTask
    from src.runtime.nodes.synthesis.evidence_selection import requirement_coverage

    source = f"def target({signature}):\n    return value\n"
    payload = _search_code_aspect(source, aspect)
    task = RetrievalTask(route="upload", query=f"target {aspect}", k=4,
                         requirement={"symbols": ["target"], "match": "definition", "aspects": [aspect]})
    evidence = [hit.evidence for hit in payload.hits]
    coverage = requirement_coverage(task, evidence, {item.id: [task.requirement_id] for item in evidence})

    assert payload.diagnostics.answerability == "covered"
    assert coverage["is_partial"] is False
    assert coverage["missing_aspects"] == []


@pytest.mark.parametrize("arguments", ["", "value", " value", "value, other"])
def test_call_prefix_fragment_does_not_require_whitespace_before_arguments(arguments):
    source = f"def target(value, other):\n    return helper({arguments})\n"
    payload = _search_code_aspect(source, "helper(")

    assert payload.diagnostics.answerability == "covered"
    assert payload.hits[0].evidence.excerpt == source.rstrip("\n")


@pytest.mark.parametrize("body", [
    "def target(np):\n    return np.array([1])\n",
    "np = service\nnp.array([1])\n",
    "def target():\n    np = service\n    return np.array([1])\n",
    "def target():\n    result = np.array([1])\n    np = service\n    return result\n",
    "def outer(np):\n    def inner():\n        return np.array([1])\n",
    "for np in services:\n    np.array([1])\n",
    "with service() as np:\n    np.array([1])\n",
    "def target[T](np: T):\n    return np.array([1])\n",
    "def target[np](value: np):\n    return np.array([1])\n",
])
def test_import_alias_is_not_applied_to_a_shadowed_name(body):
    source = "import numpy as np\n" + body
    payload = build_upload_search_tool()(query="numpy.array", requirement=_requirement(["numpy.array"], "symbol"),
                                        retriever=_source_retriever(source))

    assert payload.hits == []
    assert payload.diagnostics.answerability == "missing"


def test_shadowed_function_parameter_does_not_change_outer_import_alias():
    source = ("import numpy as np\n"
              "def shadowed(np):\n    return np.array([1])\n"
              "def preserved():\n    return np.array([2])\n")
    payload = build_upload_search_tool()(query="numpy.array", requirement=_requirement(["numpy.array"], "symbol"),
                                        retriever=_source_retriever(source))

    assert payload.diagnostics.answerability == "covered"
    assert [hit.evidence.excerpt for hit in payload.hits] == ["    return np.array([2])"]


@pytest.mark.parametrize(("prefix", "symbol", "covered"), [
    ("import numpy as np\ndef outer(np):\n", "numpy.array", True),
    ("import numpy as np\ndef outer():\n    import pandas as np\n", "numpy.array", True),
    ("import numpy as np\ndef outer():\n    import pandas as np\n", "pandas.array", False),
    ("def outer():\n    import numpy as np\n", "numpy.array", False),
])
def test_explicit_global_uses_module_binding_instead_of_enclosing_binding(prefix, symbol, covered):
    source = prefix + "    def inner():\n        global np\n        return np.array([1])\n"
    payload = build_upload_search_tool()(query=symbol, requirement=_requirement([symbol], "symbol"),
                                        retriever=_source_retriever(source))

    assert payload.diagnostics.answerability == ("covered" if covered else "missing")
    assert [hit.evidence.excerpt for hit in payload.hits] == (["        return np.array([1])"] if covered else [])


def test_assignment_visits_the_value_before_rebinding_an_import_alias():
    source = "import numpy as np\nnp = np.array([1])\nnp.array([2])\n"
    payload = build_upload_search_tool()(query="numpy.array", requirement=_requirement(["numpy.array"], "symbol"),
                                        retriever=_source_retriever(source))

    assert payload.diagnostics.answerability == "covered"
    assert [hit.evidence.excerpt for hit in payload.hits] == ["np = np.array([1])"]


def test_assignment_target_remains_searchable_by_its_source_name():
    source = "target = value\n"
    payload = build_upload_search_tool()(query="target", requirement=_requirement(["target"], "symbol"),
                                        retriever=_source_retriever(source))

    assert payload.diagnostics.answerability == "covered"
    assert [hit.evidence.excerpt for hit in payload.hits] == ["target = value"]


@pytest.mark.parametrize("shadowed", [
    "callback = lambda np: np.array([1])",
    "results = [np.array([1]) for np in services]",
])
def test_expression_scope_shadows_import_alias_without_changing_outer_scope(shadowed):
    source = f"import numpy as np\n{shadowed}\nresult = np.array([2])\n"
    payload = build_upload_search_tool()(query="numpy.array", requirement=_requirement(["numpy.array"], "symbol"),
                                        retriever=_source_retriever(source))

    assert payload.diagnostics.answerability == "covered"
    assert [hit.evidence.excerpt for hit in payload.hits] == ["result = np.array([2])"]


def test_lambdas_on_the_same_line_have_distinct_lexical_bindings():
    expression = "callbacks = (lambda np: np.array([1]), lambda value: np.array([2]))"
    source = "import numpy as np\n" + expression + "\n"
    payload = build_upload_search_tool()(query="numpy.array", requirement=_requirement(["numpy.array"], "symbol"),
                                        retriever=_source_retriever(source))

    assert payload.diagnostics.answerability == "covered"
    assert [hit.evidence.excerpt for hit in payload.hits] == [expression]


def test_notebook_import_alias_is_shared_until_a_later_cell_rebinds_it(uploaded):
    cells = ["import numpy as np\n", "result = np.array([1])\n", "np = service\nnp.array([2])\n"]
    with uploaded(cells, notebook=True) as retriever:
        payload = build_upload_search_tool()(query="numpy.array", requirement=_requirement(["numpy.array"], "symbol"),
                                            retriever=retriever)

    assert payload.diagnostics.answerability == "covered"
    assert [hit.evidence.excerpt for hit in payload.hits] == ["result = np.array([1])"]
    assert payload.hits[0].evidence.element.anchors[0].cell_index == 1


@pytest.mark.parametrize("definition", [
    "def outer[T](value: T):\n    def inner():\n        return np.array([1])\n",
    "class Wrapper[T]:\n    def method(self):\n        return np.array([1])\n",
])
def test_generic_scope_preserves_unshadowed_outer_imports(definition):
    source = "import numpy as np\n" + definition
    payload = build_upload_search_tool()(query="numpy.array", requirement=_requirement(["numpy.array"], "symbol"),
                                        retriever=_source_retriever(source))

    assert payload.diagnostics.answerability == "covered"
    assert [hit.evidence.excerpt.strip() for hit in payload.hits] == ["return np.array([1])"]


@pytest.mark.parametrize("declaration", [
    "type Factory = type(lambda np: np.array([1]))",
    "type Factory[T] = type(lambda np: np.array([1]))",
    "def target[T](value: (lambda np: np.array([1]))):\n    return value",
    "def target[T](value = lambda np: np.array([1])):\n    return value",
])
def test_annotation_and_default_expression_scopes_preserve_the_outer_alias(declaration):
    source = f"import numpy as np\n{declaration}\nresult = np.array([2])\n"
    payload = build_upload_search_tool()(query="numpy.array", requirement=_requirement(["numpy.array"], "symbol"),
                                        retriever=_source_retriever(source))

    assert payload.diagnostics.answerability == "covered"
    assert [hit.evidence.excerpt for hit in payload.hits] == ["result = np.array([2])"]


def test_unavailable_compiler_scope_does_not_claim_symbol_absence():
    source = ("from __future__ import annotations\nimport numpy as np\n"
              "def target(value: (lambda np: np.array([1]))):\n    return value\n")
    payload = build_upload_search_tool()(query="numpy.array", requirement=_requirement(["numpy.array"], "symbol"),
                                        retriever=_source_retriever(source))

    assert payload.hits == []
    assert payload.diagnostics.answerability == "unknown"
    assert payload.diagnostics.missing_requirements == []
    assert "source_symbol_analysis_unavailable" in payload.diagnostics.warnings


@pytest.mark.parametrize("declaration", [
    "def target[T: np.ndarray]():\n    pass",
    "class Target[T: np.ndarray]:\n    pass",
    "type Target[T: np.ndarray] = list[T]",
])
def test_generic_type_parameter_bound_remains_searchable(declaration):
    source = f"import numpy as np\n{declaration}\n"
    payload = build_upload_search_tool()(query="numpy.ndarray", requirement=_requirement(["numpy.ndarray"], "symbol"),
                                        retriever=_source_retriever(source))

    assert payload.diagnostics.answerability == "covered"
    assert [hit.evidence.excerpt for hit in payload.hits] == [declaration]


def test_planned_code_fragment_reaches_source_retrieval_without_losing_the_constraint():
    from src.core.contracts.boundary.graph import build_graph_state_input
    from src.runtime.nodes.planner import make_planner_node
    from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
    from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs
    from src.runtime.nodes.synthesis.evidence_selection import requirement_coverage
    from tests.core.helpers import _CapturePlannerLLM

    source = "def target(value):\n    return value is None\n"
    query = "첨부한 target 함수의 `is None` 코드 조각을 찾아 인용해줘."
    planner = _CapturePlannerLLM(PlannerOutput.model_validate({"use_retrieval": True, "tasks": [{
        "route": "upload", "query": query, "k": 4,
        "requirement": {"symbols": ["target"], "match": "definition", "aspects": ["is None"]},
    }]}))
    state = build_graph_state_input(user_input=query, retriever=_source_retriever(source))

    state.update(make_planner_node(planner, False)(state))
    state.update(make_retrieve_dispatch_node(None, build_upload_search_tool(), False)(state))

    task, = state["planner"].output.tasks
    assert task.requirement.aspects == ["is None"]
    diagnostic, = state["debug"].retrieval_diagnostics
    assert diagnostic.answerability == "covered"
    assert [hit.evidence.excerpt for hit in state["retrieval"].hit_log] == [source.rstrip("\n")]
    prepare_synthesis_state(state, limits=synthesis_excerpt_limits(normal_chars=480))
    prepared = prepare_synthesis_inputs(state=state, context=build_synthesis_context(state=state), max_turns=6)
    assert prepared.missing_requirement_ids == []
    assert [item.excerpt for item in prepared.evidence_packet] == [source.rstrip("\n")]
    assert requirement_coverage(task, prepared.evidence_packet, prepared.evidence_requirement_map)["is_partial"] is False


def _retrieve_planned_topic(retriever: UploadRetriever, query: str, aspects: list[str]) -> GraphState:
    file_id = retriever.source_documents[0].elements[0].metadata["file_id"]
    planner = _CapturePlannerLLM(PlannerOutput.model_validate({"use_retrieval": True, "tasks": [{
        "route": "upload", "query": query, "k": 4,
        "requirement": {"file_ids": [file_id], "match": "topic", "aspects": aspects},
    }]}))
    state = build_graph_state_input(user_input=query, retriever=retriever)
    state.update(make_planner_node(planner, False)(state))
    state.update(make_retrieve_dispatch_node(None, build_upload_search_tool(), False)(state))
    return state


@pytest.mark.parametrize(("query", "notebook"), [
    pytest.param("첨부한 파일의 1행·2행을 한 문장에서 설명해줘.", False, id="korean-lines"),
    pytest.param("Explain line 1 and line 2 of the uploaded file.", False, id="english-lines"),
    pytest.param("Explain the source range from line 1 through line 2.", False, id="line-range"),
    pytest.param("첨부한 노트북 첫 번째 셀의 1행·2행을 설명해줘.", True, id="notebook-cell"),
])
def test_planned_source_locations_remain_query_context_without_literal_requirements(uploaded, query, notebook):
    """A valid provider plan preserves location requests without demanding those words in source."""
    source = "x = 1\ny = 2\n"
    with uploaded([source] if notebook else source, notebook=notebook) as retriever:
        state = _retrieve_planned_topic(retriever, query, [])

    task, = state["planner"].output.tasks
    assert task.query == query
    assert task.requirement.aspects == []
    diagnostic, = state["debug"].retrieval_diagnostics
    assert diagnostic.answerability == "covered"
    assert diagnostic.missing_requirements == []
    evidence, = [hit.evidence for hit in state["retrieval"].hit_log]
    assert evidence.excerpt.rstrip("\n") == "x = 1\ny = 2"
    assert evidence.excerpt == source[evidence.selection.start:evidence.selection.end]
    coverage = requirement_coverage(task, [evidence], {evidence.id: [task.requirement_id]})
    assert coverage["is_partial"] is False
    assert coverage["missing_aspects"] == []
    if notebook:
        assert evidence.element.anchors[0].cell_index == 0
        assert evidence.element.anchors[0].cell_id == "source-0"


@pytest.mark.parametrize(("source", "answerability", "missing_aspects"), [
    pytest.param("x = 1\ny = 2\n", "partial", ["1행"], id="literal-absent"),
    pytest.param('label = "1행"\ny = 2\n', "covered", [], id="literal-present"),
])
def test_planned_location_word_as_literal_still_requires_source_occurrence(uploaded, source, answerability, missing_aspects):
    """The same location-looking text is binding when requested as an actual source string."""
    query = "첨부한 파일에서 원문 문자열 `1행`을 찾아 인용해줘."
    with uploaded(source) as retriever:
        state = _retrieve_planned_topic(retriever, query, ["1행"])

    task, = state["planner"].output.tasks
    assert task.requirement.aspects == ["1행"]
    diagnostic, = state["debug"].retrieval_diagnostics
    assert diagnostic.answerability == answerability
    assert diagnostic.missing_requirements == ([f"file:{task.requirement.file_ids[0]}:aspect:1행"] if missing_aspects else [])
    evidence = [hit.evidence for hit in state["retrieval"].hit_log]
    assert evidence
    coverage = requirement_coverage(task, evidence, {item.id: [task.requirement_id] for item in evidence})
    assert coverage["is_partial"] is bool(missing_aspects)
    assert coverage["missing_aspects"] == missing_aspects


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
        candidates = retriever.search("메서드를 발췌", k=1, mode="baseline")
        assert all("summarize_rows" not in hit.evidence.excerpt for hit in candidates)
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
    prepare_synthesis_state(state, limits=synthesis_excerpt_limits(normal_chars=480))
    prepared = prepare_synthesis_inputs(state=state, context=build_synthesis_context(state=state), max_turns=6)
    assert {item.excerpt for item in prepared.evidence_packet} == {definition, example}
    assert all(len(item.excerpt) <= 480 for item in prepared.evidence_packet)
