"""Multiple uploads retain file boundaries through retrieval and old citations."""

from __future__ import annotations

import hashlib
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

    def record(name, content):
        from src.core.uploads import UploadRecord

        file_id = uuid4().hex
        data = content.encode("utf-8")
        path = tmp_path / "uploads" / "multi" / uuid4().hex / name
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        return UploadRecord(file_id=file_id, name=name, path=str(path), size_bytes=len(data),
                            content_hash="sha256:" + hashlib.sha256(data).hexdigest(),
                            source_uri=f"upload://multi/{file_id}")

    @contextmanager
    def indexed(files):
        from src.infra.tools.local_rag.uploads import build_upload_retriever

        handle = build_upload_retriever(files, session_id="multi", generation=uuid4().hex,
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


def test_scoped_search_preserves_every_file(uploads):
    """Uneven file sizes cannot exhaust another file's candidates."""
    record, indexed = uploads
    contents = {"large.py": "value = 1\n" * 300, "small.py": "other_value = 2\n", "third.py": "value = 3\n"}
    files = [record(name, content) for name, content in contents.items()]
    requirement = RetrievalRequirement(file_ids=[file.file_id for file in files])
    search = build_upload_search_tool()
    with indexed(files) as handle:
        payload = search(query="compare values", k=1, retriever=handle.retriever, requirement=requirement)
        hits = payload.hits
        assert {hit.evidence.snapshot.title for hit in hits} == set(contents)
        for hit in hits:
            source = contents[hit.evidence.snapshot.title]
            assert hit.evidence.element.text == source
            assert hit.evidence.excerpt
            assert hit.evidence.excerpt == source[hit.evidence.selection.start:hit.evidence.selection.end]
        assert payload.diagnostics.answerability == "covered"
        assert payload.diagnostics.missing_requirements == []


def test_function_search_across_attachments_does_not_require_each_file_to_define_it(uploads):
    """An implementation and a caller have different roles; explicit comparisons still need both."""
    record, indexed = uploads
    implementation = record("billing.py", "def calculate_total(amount, discount=0.10):\n    return amount * (1 - discount)\n")
    example = record("example.py", "from billing import calculate_total\nresult = calculate_total(30000)\n")
    requirement = RetrievalRequirement(symbols=["calculate_total"], match="definition")
    with indexed([implementation, example]) as handle:
        search = build_upload_search_tool()
        found = search(query="첨부의 calculate_total 함수에서 기본 할인율", retriever=handle.retriever,
                       requirement=requirement)
        comparison = search(query="두 파일 각각에 있는 calculate_total 정의 비교", retriever=handle.retriever,
                            requirement=requirement.model_copy(update={"file_ids": [implementation.file_id, example.file_id]}))
    assert found.diagnostics.answerability == "covered"
    assert found.diagnostics.missing_requirements == []
    assert [hit.evidence.snapshot.title for hit in found.hits] == ["billing.py"]
    assert "discount=0.10" in found.hits[0].evidence.excerpt
    assert comparison.diagnostics.answerability == "partial"
    assert comparison.diagnostics.missing_requirements == [f"file:{example.file_id}:calculate_total"]
