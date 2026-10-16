"""Multiple uploads retain file boundaries through retrieval and old citations."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

from src.core.planner_schema import RetrievalRequirement
from src.infra.tools.local_rag import build_upload_search_tool


class LengthEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.0]


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: LengthEmbeddings())

    def record(name, content, *, file_id=None, session="multi"):
        from src.core.uploads import UploadRecord

        file_id = file_id or uuid4().hex
        data = content.encode("utf-8")
        path = tmp_path / "uploads" / session / uuid4().hex / name
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        return UploadRecord(file_id=file_id, name=name, path=str(path), size_bytes=len(data),
                            content_hash="sha256:" + hashlib.sha256(data).hexdigest(),
                            source_uri=f"upload://{session}/{file_id}")

    @contextmanager
    def indexed(files, *, session="multi", generation=None):
        from src.infra.tools.local_rag.uploads import build_upload_retriever

        handle = build_upload_retriever(files, session_id=session, generation=generation or uuid4().hex,
                                       api_key="test-key")
        try:
            yield handle
        finally:
            handle.cleanup()

    return record, indexed


def test_same_content_in_different_files_has_distinct_citations(uploads):
    """Equal text and overlapping element IDs still resolve to each file's original source."""
    record, indexed = uploads
    first = record("first.py", "def load():\n    return 1\n")
    second = record("second.py", "def load():\n    return 1\n")
    with indexed([first, second]) as handle:
        payload = build_upload_search_tool()(query="load definition", retriever=handle.retriever,
                                            requirement=RetrievalRequirement(symbols=["load"], match="definition"))
        direct = handle.retriever.vectorstore.similarity_search_with_score("load")
        assert len(direct) == 2
        assert len(handle.retriever.source_documents) == 2
    hits = payload.hits
    assert {hit.evidence.snapshot.title for hit in hits} == {"first.py", "second.py"}
    assert {hit.evidence.element.metadata["file_id"] for hit in hits} == {first.file_id, second.file_id}
    assert len({hit.evidence.id for hit in hits}) == 2
    assert all(hit.evidence.excerpt == "def load():\n    return 1" for hit in hits)


@pytest.mark.parametrize("symbols", [[], ["load"]])
def test_explicit_file_scope_applies_to_semantic_and_exact_search(uploads, symbols):
    """A requested file scope prevents another active file from supplying the evidence."""
    record, indexed = uploads
    first = record("first.py", "def load():\n    return 'first'\n")
    second = record("second.py", "def load():\n    return 'second'\n")
    with indexed([first, second]) as handle:
        payload = build_upload_search_tool()(query="load", retriever=handle.retriever,
            requirement=RetrievalRequirement(symbols=symbols, match="definition" if symbols else "topic", file_ids=[second.file_id]))
    hits = payload.hits
    assert hits
    assert payload.diagnostics.answerability == "covered"
    assert {hit.evidence.snapshot.title for hit in hits} == {"second.py"}
    assert {hit.evidence.element.metadata["file_id"] for hit in hits} == {second.file_id}
    assert all("return 'second'" in hit.evidence.excerpt for hit in hits)

























