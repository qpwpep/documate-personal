"""Multiple uploads retain file boundaries through retrieval and old citations."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

from src.core.evidence import parse_search_hits
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
        direct = handle.retriever.invoke("load")
        assert len(direct) == 2
        assert len(handle.retriever.source_documents) == 2
    hits = parse_search_hits(payload)
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
            requirement=RetrievalRequirement(symbols=symbols, match="definition", file_ids=[second.file_id]))
    hits = parse_search_hits(payload)
    assert hits
    assert {hit.evidence.snapshot.title for hit in hits} == {"second.py"}


def test_unknown_file_scope_is_rejected_without_unscoped_fallback(uploads):
    """A foreign or stale file ID yields an error instead of searching all active files."""
    record, indexed = uploads
    with indexed([record("first.py", "value = 1\n")]) as handle:
        payload = build_upload_search_tool()(query="value", retriever=handle.retriever,
            requirement=RetrievalRequirement(file_ids=["foreign-file"]))
    assert payload["hits"] == []
    assert payload["diagnostics"]["status"] == "error"
    assert payload["diagnostics"]["error_code"] == "UPLOAD_FILE_SCOPE_INVALID"


def test_explicit_multi_file_scope_reserves_candidates_for_every_file(uploads):
    """A larger file cannot consume all candidates in an explicit multi-file comparison."""
    record, indexed = uploads
    first = record("large.py", "value = 1\n" * 300)
    second = record("small.py", "other_value = 2\n")
    with indexed([first, second]) as handle:
        payload = build_upload_search_tool()(query="compare values", k=1, retriever=handle.retriever,
            requirement=RetrievalRequirement(file_ids=[first.file_id, second.file_id]))
    assert {hit.evidence.snapshot.title for hit in parse_search_hits(payload)} == {"large.py", "small.py"}
    assert payload["diagnostics"]["answerability"] == "covered"
    assert payload["diagnostics"]["missing_requirements"] == []


def test_scoped_search_embeds_query_once_per_invocation(uploads, monkeypatch):
    """Each search embeds once while preserving every requested file, even when k is smaller."""
    class RecordingEmbeddings(LengthEmbeddings):
        def __init__(self):
            self.queries = []

        def embed_query(self, text):
            self.queries.append(text)
            return super().embed_query(text)

    embeddings = RecordingEmbeddings()
    monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: embeddings)
    record, indexed = uploads
    files = [record(f"file-{index}.py", f"value = {index}\n") for index in range(3)]
    requirement = RetrievalRequirement(file_ids=[file.file_id for file in files])
    search = build_upload_search_tool()
    with indexed(files) as handle:
        for _ in range(2):
            payload = search(query="compare values", k=1, retriever=handle.retriever, requirement=requirement)
            assert {hit.evidence.snapshot.title: hit.evidence.excerpt for hit in parse_search_hits(payload)} == {
                f"file-{index}.py": f"value = {index}" for index in range(3)
            }
            assert payload["diagnostics"]["answerability"] == "covered"
            assert payload["diagnostics"]["missing_requirements"] == []
    assert embeddings.queries == ["compare values", "compare values"]


def test_scoped_candidates_preserve_filters_reserved_sources_and_raw_distances(uploads):
    """Scoped retrieval reserves each source before filling the budget by raw distance within both filters."""
    from src.core.evidence import EvidenceRef

    def notebook(cells):
        return json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
            {"id": f"cell-{index}", "cell_type": "code", "source": source, "metadata": {},
             "execution_count": None, "outputs": []} for index, source in enumerate(cells)
        ]})

    record, indexed = uploads
    first = record("first.ipynb", notebook([
        "value = 0\n", "other = 0\n", "value = 111\n", "value = 11111\n", "value = 11111111111\n",
    ]))
    second = record("second.ipynb", notebook([
        "value = 0\n", "value = 2\n", "value = 222222\n", "value = 2222222222222\n",
    ]))
    excluded = record("excluded.ipynb", notebook(["value = 0\n", "value\n"]))
    with indexed([first, second, excluded]) as handle:
        rows = handle.retriever.vectorstore.similarity_search_with_score(
            query="value", k=4, file_ids=[first.file_id, second.file_id],
            filter={"cell_index": {"$gte": 1}}, where_document={"$contains": "value"},
        )
    assert [(doc.page_content, score) for doc, score in rows] == [
        ("value = 111\n", 49.0), ("value = 2\n", 25.0),
        ("value = 11111\n", 81.0), ("value = 222222\n", 100.0),
    ]
    evidence = [EvidenceRef.model_validate_json(doc.metadata["evidence_ref"]) for doc, _score in rows]
    assert [(item.snapshot.title, item.element.metadata["file_id"], item.element.anchors[0].cell_index, item.excerpt)
            for item in evidence] == [
        ("first.ipynb", first.file_id, 2, "value = 111\n"),
        ("second.ipynb", second.file_id, 1, "value = 2\n"),
        ("first.ipynb", first.file_id, 3, "value = 11111\n"),
        ("second.ipynb", second.file_id, 2, "value = 222222\n"),
    ]


def test_file_coverage_does_not_establish_an_unverified_code_aspect(uploads):
    """File-scoped semantic candidates do not prove additional explicit code constraints."""
    record, indexed = uploads
    first = record("first.py", "value = 1\n")
    with indexed([first]) as handle:
        payload = build_upload_search_tool()(query="value axis", retriever=handle.retriever,
            requirement=RetrievalRequirement(file_ids=[first.file_id], aspects=["axis"]))
    assert payload["diagnostics"]["answerability"] == "unknown"


def test_import_alias_does_not_leak_to_another_file(uploads):
    """An alias imported in one file cannot turn another file's unrelated call into that API."""
    record, indexed = uploads
    first = record("imports.py", "from itertools import chain as combine\n")
    second = record("other.py", "rows = combine(left, right)\n")
    with indexed([first, second]) as handle:
        payload = build_upload_search_tool()(query="itertools.chain", retriever=handle.retriever,
            requirement=RetrievalRequirement(symbols=["itertools.chain"]))
    hits = parse_search_hits(payload)
    assert {hit.evidence.snapshot.title for hit in hits} == {"imports.py"}
    assert all("rows =" not in hit.evidence.excerpt for hit in hits)


def test_notebook_aliases_cross_cells_but_not_files(uploads):
    """Multi-file indexing preserves notebook cell execution context within its own source."""
    record, indexed = uploads
    cells = ["from itertools import chain as combine\n", "rows = list(combine(left, right))\n"]
    notebook = json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
        {"id": f"cell-{i}", "cell_type": "code", "source": text, "metadata": {},
         "execution_count": None, "outputs": []} for i, text in enumerate(cells)
    ]})
    first = record("notebook.ipynb", notebook)
    second = record("other.py", "combine(values)\n")
    with indexed([first, second]) as handle:
        payload = build_upload_search_tool()(query="itertools.chain", retriever=handle.retriever,
            requirement=RetrievalRequirement(symbols=["itertools.chain"]))
    hits = parse_search_hits(payload)
    assert [hit.evidence.excerpt for hit in hits] == [cells[1].rstrip("\n")]
    assert hits[0].evidence.element.anchors[0].cell_index == 1


def test_old_generation_cleanup_preserves_new_search_and_old_citation(uploads):
    """Replacement has a new snapshot while old index disposal cannot damage new or returned data."""
    record, indexed = uploads
    first = record("code.py", "value = 1\n")
    replacement = record("code.py", "value = 2\n", file_id=first.file_id)
    with indexed([first]) as previous:
        payload = build_upload_search_tool()(query="value", retriever=previous.retriever)
        old = parse_search_hits(payload)[0].evidence
        frozen = old.model_dump(mode="json")
        with indexed([replacement]) as current:
            previous.cleanup()
            current_payload = build_upload_search_tool()(query="value", retriever=current.retriever)
            new = parse_search_hits(current_payload)[0].evidence
            assert new.snapshot.document_id == old.snapshot.document_id
            assert new.snapshot.snapshot_id != old.snapshot.snapshot_id
            assert new.excerpt == "value = 2"
    assert old.model_dump(mode="json") == frozen
    assert old.excerpt == "value = 1"


def test_changed_staged_bytes_are_rejected_before_indexing(uploads):
    """A file changed after validation cannot create an index for the wrong approved revision."""
    from pathlib import Path

    record, indexed = uploads
    file = record("code.py", "value = 1\n")
    Path(file.path).write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        with indexed([file]):
            pass


def test_explicit_scope_reports_the_file_missing_a_required_symbol(uploads):
    """Finding a definition in only one requested file cannot cover a two-file comparison."""
    record, indexed = uploads
    first = record("first.py", "def load():\n    return 1\n")
    second = record("second.py", "def store():\n    return 2\n")
    with indexed([first, second]) as handle:
        payload = build_upload_search_tool()(query="compare load definitions", retriever=handle.retriever,
            requirement=RetrievalRequirement(symbols=["load"], match="definition",
                                             file_ids=[first.file_id, second.file_id]))
    assert payload["diagnostics"]["answerability"] == "partial"
    assert payload["diagnostics"]["missing_requirements"] == [f"file:{second.file_id}:load"]
    assert {hit.evidence.snapshot.title for hit in parse_search_hits(payload)} == {"first.py"}


def test_failed_candidate_index_is_removed_without_harming_active_search(uploads, monkeypatch):
    """A failure after one file was indexed rolls back its candidate collection and permits a clean retry."""
    class FailSecondEmbedding(LengthEmbeddings):
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("embedding boundary failed")
            return super().embed_documents(texts)

    record, indexed = uploads
    first = record("first.py", "first = 1\n")
    second = record("second.py", "second = 2\n")
    with indexed([first]) as active:
        monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: FailSecondEmbedding())
        with pytest.raises(RuntimeError, match="second.py: embedding or vector insertion failed"):
            with indexed([first, second], generation="failed-candidate"):
                pass
        assert [doc.page_content for doc in active.retriever.invoke("first")] == ["first = 1\n"]
        monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: LengthEmbeddings())
        with indexed([first, second], generation="failed-candidate") as retried:
            assert len(retried.retriever.vectorstore.get()["ids"]) == 2


def test_invalid_notebook_identifies_the_failed_file(uploads):
    """Batch parsing failures name the file that must be fixed before retrying."""
    record, indexed = uploads
    first = record("valid.py", "value = 1\n")
    invalid = record("broken.ipynb", "{ not json }")
    with pytest.raises(ValueError, match="broken.ipynb:"):
        with indexed([first, invalid]):
            pass


def test_source_resolver_enforces_scope_when_called_directly(uploads):
    """The source resolver does not return definitions outside its explicit file constraint."""
    from src.infra.tools.local_rag.requirements import resolve_source_requirement

    record, indexed = uploads
    first = record("first.py", "def load():\n    return 1\n")
    second = record("second.py", "def load():\n    return 2\n")
    with indexed([first, second]) as handle:
        result = resolve_source_requirement(source_documents=handle.retriever.source_documents,
            candidate_rows=[], requirement=RetrievalRequirement(symbols=["load"], match="definition", file_ids=[second.file_id]))
    assert {hit.evidence.snapshot.title for hit in result.hits} == {"second.py"}
