from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings

from src.core.evidence import parse_search_hits
from src.infra.chunking import chunk_python_text
from src.infra.tools.local_rag import build_temp_retriever, build_upload_search_tool


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
        handle = build_temp_retriever(str(path), api_key="test-key")
        try:
            yield handle.retriever
        finally:
            handle.cleanup()

    return create


def _requirement(symbols, match="definition"):
    from src.core.planner_schema import RetrievalRequirement
    return RetrievalRequirement(symbols=symbols, match=match)


def test_legacy_definition_request_does_not_quote_a_comment_or_a_call(uploaded):
    """A requested function definition is missing even when its name occurs in a comment and call."""
    source = "# archive_orders is supplied by another package\narchive_orders(items)\n\ndef archive_users():\n    return []\n"
    with uploaded(source) as retriever:
        payload = build_upload_search_tool()(query="archive_orders 함수 정의를 원문 그대로 발췌해줘", retriever=retriever)
    assert payload["hits"] == []
    assert payload["diagnostics"]["answerability"] == "missing"
    assert payload["diagnostics"]["missing_requirements"] == ["archive_orders"]


@pytest.mark.parametrize("query, name", [
    ("Extract function collect_records verbatim", "collect_records"),
    ("Quote the rebalance function definition", "rebalance"),
    ("`normalizeRow` 함수 원문을 발췌해줘", "normalizeRow"),
])
def test_legacy_named_definition_phrases_keep_the_exact_requested_name(uploaded, query, name):
    """Definition requests identify the code name rather than English instruction words."""
    with uploaded(f"def {name}():\n    return 1\n") as retriever:
        payload = build_upload_search_tool()(query=query, retriever=retriever)
    assert payload["diagnostics"]["answerability"] == "covered"
    assert payload["diagnostics"]["missing_requirements"] == []
    assert [hit.evidence.excerpt for hit in parse_search_hits(payload)] == [f"def {name}():\n    return 1"]


def test_complete_method_definition_is_found_outside_vector_top_k_with_exact_offsets(uploaded):
    """Exact source lookup retrieves a complete decorated method regardless of vector candidate limits."""
    method = "    @staticmethod\r\n    async def summarize_rows(rows):\r\n        label = '한글'\r\n" + "        rows = list(rows)\r\n" * 35 + "        return rows\r\n"
    source = "unrelated = 1\r\n" + "# other setup\r\n" * 90 + "class ReportBuilder:\r\n" + method + "\r\ndef other():\r\n    return None\r\n"
    with uploaded(source) as retriever:
        candidates = retriever.vectorstore.similarity_search_with_score("메서드를 발췌", k=1)
        assert all("summarize_rows" not in doc.page_content for doc, _score in candidates)
        payload = build_upload_search_tool()(query="메서드를 발췌", k=1, retriever=retriever,
                                            requirement=_requirement(["ReportBuilder.summarize_rows"]))
    hits = parse_search_hits(payload)
    assert payload["diagnostics"]["answerability"] == "covered"
    assert [hit.evidence.excerpt for hit in hits] == [method.rstrip("\r\n")]
    assert hits[0].evidence.excerpt == source[hits[0].evidence.selection.start:hits[0].evidence.selection.end]
    assert hits[0].evidence.element.text == source


@pytest.mark.parametrize("match, expected", [("symbol", "covered"), ("definition", "missing")])
def test_library_method_usage_is_distinct_from_a_local_definition(uploaded, match, expected):
    """A call to a library method covers a usage request but cannot stand in for its definition."""
    with uploaded("clean = frame.fillna(0)\n") as retriever:
        payload = build_upload_search_tool()(query="fillna", retriever=retriever,
                                            requirement=_requirement(["fillna"], match))
    assert payload["diagnostics"]["answerability"] == expected
    assert bool(payload["hits"]) == (expected == "covered")


@pytest.mark.parametrize("body, expected", [
    ("return frame.fillna(0, axis=1)", "covered"),
    ("# axis is not passed here\n    return frame.fillna(0)", "unknown"),
])
def test_code_aspects_require_syntax_in_the_selected_definition(uploaded, body, expected):
    """An aspect present only in a comment cannot establish an actual parameter or code operation."""
    requirement = _requirement(["clean_frame"])
    requirement.aspects = ["axis"]
    with uploaded(f"def clean_frame(frame):\n    {body}\n") as retriever:
        payload = build_upload_search_tool()(query="clean_frame axis", retriever=retriever, requirement=requirement)
    assert payload["diagnostics"]["answerability"] == expected


def test_code_definition_does_not_establish_an_unrecorded_library_version(uploaded):
    """A matching definition alone cannot establish an explicit dependency version constraint."""
    requirement = _requirement(["normalize_index"])
    requirement.version = "2.1"
    with uploaded("def normalize_index(frame):\n    return frame.reset_index()\n") as retriever:
        payload = build_upload_search_tool()(query="normalize_index", retriever=retriever, requirement=requirement)
    assert payload["diagnostics"]["answerability"] == "unknown"


def test_multiple_required_definitions_report_only_the_missing_symbol(uploaded):
    """Partially answerable requests keep the available definition and identify the absent one."""
    with uploaded("def load_records():\n    return []\n") as retriever:
        payload = build_upload_search_tool()(query="두 함수를 발췌", retriever=retriever,
                                            requirement=_requirement(["load_records", "persist_records"]))
    assert payload["diagnostics"]["answerability"] == "partial"
    assert payload["diagnostics"]["missing_requirements"] == ["persist_records"]
    assert [hit.evidence.excerpt for hit in parse_search_hits(payload)] == ["def load_records():\n    return []"]


def test_notebook_definitions_keep_each_cell_identity(uploaded):
    """Repeated definitions in different notebook cells remain separate traceable source selections."""
    cells = ["def transform_batch(values):\n    return values\n", "def transform_batch(values):\n    return list(values)\n"]
    with uploaded(cells, notebook=True) as retriever:
        payload = build_upload_search_tool()(query="transform_batch 정의", retriever=retriever,
                                            requirement=_requirement(["transform_batch"]))
    hits = parse_search_hits(payload)
    assert [hit.evidence.element.anchors[0].cell_index for hit in hits] == [0, 1]
    assert [hit.evidence.excerpt for hit in hits] == [cell.rstrip("\n") for cell in cells]
    assert len({hit.evidence.id for hit in hits}) == 2


def test_notebook_import_alias_identifies_usage_in_a_later_cell(uploaded):
    """A preserved import alias connects a qualified API requirement to its later notebook call."""
    cells = ["from itertools import chain as combine\n", "rows = list(combine(left, right))\n"]
    with uploaded(cells, notebook=True) as retriever:
        payload = build_upload_search_tool()(query="itertools.chain 사용", retriever=retriever,
                                            requirement=_requirement(["itertools.chain"], "symbol"))
    hits = parse_search_hits(payload)
    assert payload["diagnostics"]["answerability"] == "covered"
    assert [hit.evidence.excerpt for hit in hits] == [cells[1].rstrip("\n")]
    assert hits[0].evidence.element.anchors[0].cell_index == 1


def test_absence_without_a_complete_source_registry_remains_unknown():
    """A vector candidate that lacks the target cannot prove the target is absent from the upload."""
    indexed = chunk_python_text(path="uploads/candidates/code.py", text="def nearby():\n    return 1\n",
                                chunk_size=800, chunk_overlap=100)

    class CandidateStore:
        def similarity_search_with_score(self, query, k=4):
            return [(indexed.hydrate(indexed.chunks[0]), 0.1)]

    payload = build_upload_search_tool()(query="read_events 정의", retriever=SimpleNamespace(vectorstore=CandidateStore()),
                                        requirement=_requirement(["read_events"]))
    assert payload["hits"] == []
    assert payload["diagnostics"]["answerability"] == "unknown"
    assert payload["diagnostics"]["missing_requirements"] == []
    assert payload["diagnostics"]["candidate_count"] == 1


def test_unparseable_notebook_cell_prevents_an_absence_claim(uploaded):
    """Unavailable AST analysis is reported as unknown instead of proving a missing definition."""
    with uploaded(["%time process_events()\n"], notebook=True) as retriever:
        payload = build_upload_search_tool()(query="process_events 정의", retriever=retriever,
                                            requirement=_requirement(["process_events"]))
    assert payload["hits"] == []
    assert payload["diagnostics"]["answerability"] == "unknown"
    assert payload["diagnostics"]["missing_requirements"] == []


def test_source_lookup_remains_isolated_between_uploaded_sessions(uploaded):
    """Exact source lookup uses the current retriever and never another session's definitions."""
    with uploaded("def private_aggregate():\n    return 42\n", session="alpha") as first:
        with uploaded("def public_aggregate():\n    return 7\n", session="beta") as second:
            found = build_upload_search_tool()(query="private_aggregate", retriever=first,
                                              requirement=_requirement(["private_aggregate"]))
            absent = build_upload_search_tool()(query="private_aggregate", retriever=second,
                                               requirement=_requirement(["private_aggregate"]))
    assert found["diagnostics"]["answerability"] == "covered"
    assert absent["hits"] == []
    assert absent["diagnostics"]["answerability"] == "missing"
