from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

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
