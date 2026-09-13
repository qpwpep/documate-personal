"""Document upload transactions through HTTP, real vector search and graph citations."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import time
from zipfile import ZipFile

import pytest

from src.core.answer_schema import export_answer_text
from src.core.documents import DocumentElement, ParsedDocument, SourceAnchor, TableCell, TableData, build_snapshot
from src.infra.document_ingestion import IngestionError
from tests.web.test_multi_upload_api import LocalEmbeddings, api, answer, context, final_response, manifest, staged, sync, sync_body


class LocalDocumentConverter:
    """Replace only the external document conversion boundary with retained structured sources."""

    def __init__(self):
        self.failure: IngestionError | None = None
        self.closed = False

    def convert_many(self, files, *, policy, workspace, cache_dir=None, deadline=None):
        if self.failure is not None:
            raise self.failure
        documents = []
        for file in files:
            raw = Path(file.path).read_bytes()
            assert Path(file.name).suffix in {".pdf", ".docx"}, "Native code must retain its existing parser"
            pdf = file.name.endswith(".pdf")
            value = "70" if b"value=70" in raw else "7"
            warnings = ["OCR extraction may contain recognition errors."] if pdf else []
            snapshot = build_snapshot(
                source_uri=file.source_uri, title=file.name, source_type="upload", content=raw,
                media_type=("application/pdf" if pdf else
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                parser="docling", parser_version="test-1", parser_config=policy.extraction_options(),
                quality_issues=warnings,
            )
            if pdf:
                page_one = SourceAnchor(kind="page", page_no=1, bbox=(0.1, 0.2, 0.9, 0.8), precision="element")
                page_two = SourceAnchor(kind="page", page_no=2, precision="page")
                elements = [DocumentElement(
                    element_id="value-table", kind="table", heading_path=["Values"],
                    anchors=[page_one, page_two], table=TableData(cells=[
                        TableCell(cell_id="title", row=0, col=0, col_span=2, text="Values", is_header=True,
                                  anchors=[page_one]),
                        TableCell(cell_id="label", row=1, col=0, text="PDF value", anchors=[page_two]),
                        TableCell(cell_id="value", row=1, col=1, text=value, anchors=[page_two]),
                    ]),
                )]
            else:
                elements = [DocumentElement(element_id="body", kind="paragraph", text="DOCX value is 9.",
                                            heading_path=["Values"])]
            documents.append(ParsedDocument(snapshot=snapshot, elements=elements))
        return tuple(documents)

    def close(self):
        self.closed = True


@pytest.fixture
def document_api(api):
    api.settings.docling_enabled = True
    api.converter = LocalDocumentConverter()
    api.client.app.state.upload_service.converter = api.converter
    return api


def pdf_bytes(value=7):
    return f"%PDF-1.7\n%value={value}\n%%EOF".encode("ascii")


def docx_bytes():
    result = BytesIO()
    with ZipFile(result, "w") as archive:
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>DOCX value is 9.</w:t></w:r></w:p></w:body></w:document>')
    return result.getvalue()


def owned_files(api):
    root = api.root / "uploads" / "session-a" / "objects"
    return {path: path.read_bytes() for path in root.glob("*/*/*") if path.is_file()}


def evidence_by_name(result):
    return {citation.evidence.snapshot.title: citation.evidence for citation in result.citations}


def test_mixed_http_uploads_preserve_native_code_structured_table_pages_and_docx_citations(document_api):
    """Mixed attachments reach the real graph with immutable source ranges, table selections and page locations."""
    api = document_api
    current = sync(api, add=[staged(api, "sample.py", "native_value = 3\n"),
                             staged(api, "report.pdf", pdf_bytes()),
                             staged(api, "guide.docx", docx_bytes())])["manifest"]

    result = answer(api, uploads=context(current))

    sources = evidence_by_name(result)
    assert set(sources) == {"sample.py", "report.pdf", "guide.docx"}
    assert {item.element.metadata["file_id"] for item in sources.values()} == {file["file_id"] for file in current["files"]}
    assert sources["sample.py"].snapshot.parser == "python-ast"
    assert sources["sample.py"].element.text == "native_value = 3\n"
    table = sources["report.pdf"]
    assert table.element.text == ""
    assert set(table.selection.cell_ids) == {"title", "label", "value"}
    assert table.excerpt == "Values\nPDF value | 7"
    assert [(anchor.page_no, anchor.bbox) for anchor in table.element.anchors] == [(1, (0.1, 0.2, 0.9, 0.8)), (2, None)]
    assert table.element.table.cells[0].col_span == 2
    assert sources["guide.docx"].excerpt == "DOCX value is 9."
    assert "OCR extraction may contain recognition errors." in export_answer_text(result, include_sources=True)
    assert all(check.support_status == "exact_match" for check in result.checks)
    assert manifest(api) == current


def test_document_replacement_and_removal_preserve_prior_answers_and_release_superseded_originals(document_api):
    """Replacing and deleting a PDF changes future retrieval while earlier cell citations retain their source version."""
    api = document_api
    current = sync(api, add=[staged(api, "sample.py", "native_value = 3\n"),
                             staged(api, "report.pdf", pdf_bytes())])["manifest"]
    previous = answer(api, uploads=context(current))
    previous_json = previous.model_dump(mode="json")
    old_sources = owned_files(api)
    pdf = next(file for file in current["files"] if file["name"] == "report.pdf")
    replacement = {**staged(api, "report.pdf", pdf_bytes(70)), "replace_file_id": pdf["file_id"]}

    updated = sync(api, add=[replacement])["manifest"]
    latest = evidence_by_name(answer(api, uploads=context(updated)))["report.pdf"]

    prior = evidence_by_name(previous)["report.pdf"]
    assert latest.element.metadata["file_id"] == prior.element.metadata["file_id"] == pdf["file_id"]
    assert latest.snapshot.source_uri == prior.snapshot.source_uri
    assert latest.snapshot.snapshot_id != prior.snapshot.snapshot_id
    assert latest.excerpt == "Values\nPDF value | 70"
    assert prior.excerpt == "Values\nPDF value | 7"
    assert all(not path.exists() for path, raw in old_sources.items() if raw == pdf_bytes())
    replacement_originals = owned_files(api)

    removed = sync(api, remove=[pdf["file_id"]])["manifest"]

    assert all(not path.exists() for path, raw in replacement_originals.items() if raw == pdf_bytes(70))
    assert set(evidence_by_name(answer(api, uploads=context(removed)))) == {"sample.py"}
    assert previous.model_dump(mode="json") == previous_json


@pytest.mark.parametrize("code,status", [("DOCUMENT_PARTIAL_CONVERSION", 422), ("DOCUMENT_PROCESSING_TIMEOUT", 504)])
def test_typed_conversion_failure_rolls_back_whole_batch_and_keeps_committed_sources(document_api, code, status):
    """A partial conversion or timeout never commits any candidate source or removes the current searchable set."""
    api = document_api
    current = sync(api, add=[staged(api, "existing.py", "existing_value = 3\n"),
                             staged(api, "report.pdf", pdf_bytes())])["manifest"]
    previous_files = owned_files(api)
    api.converter.failure = IngestionError(code, "문서 변환 실패", file_name="broken.docx",
                                          retryable=code == "DOCUMENT_PROCESSING_TIMEOUT")
    additions = [staged(api, "new.py", "new_value = 4\n"), staged(api, "broken.docx", docx_bytes())]

    response = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(current, add=additions))

    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert owned_files(api) == previous_files
    assert manifest(api) == current
    assert set(evidence_by_name(answer(api, uploads=context(current)))) == {"existing.py", "report.pdf"}


def test_embedding_failure_after_conversion_preserves_the_current_document_index(document_api):
    """An embedding outage after successful conversion leaves committed sources and their bytes unchanged."""
    api = document_api
    current = sync(api, add=[staged(api, "report.pdf", pdf_bytes())])["manifest"]
    previous_files = owned_files(api)

    def unavailable():
        raise ConnectionError("Simulated embedding outage")

    api.controls.before_embedding = unavailable
    response = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(
        current, add=[staged(api, "new.docx", docx_bytes())]))

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "UPLOAD_INDEX_FAILED"
    assert owned_files(api) == previous_files
    assert manifest(api) == current
    assert evidence_by_name(answer(api, uploads=context(current)))["report.pdf"].excerpt == "Values\nPDF value | 7"


def test_document_transactions_replay_without_conversion_and_reject_stale_revisions(document_api):
    """An operation replay uses its committed result while stale new document changes are rejected."""
    api = document_api
    empty = manifest(api)
    body = sync_body(empty, add=[staged(api, "report.pdf", pdf_bytes())])
    first = api.client.post("/sessions/session-a/uploads/sync", json=body)
    assert first.status_code == 200, first.text
    api.converter.failure = IngestionError("DOCUMENT_PROCESSING_TIMEOUT", "Must not convert replay")

    replay = api.client.post("/sessions/session-a/uploads/sync", json=body)
    stale = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(
        empty, add=[staged(api, "guide.docx", docx_bytes())]))

    assert replay.status_code == 200 and replay.json() == first.json()
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "UPLOAD_REVISION_CONFLICT"
    assert manifest(api) == first.json()["manifest"]


def test_index_rebuilds_reuse_document_embeddings_but_new_source_bytes_require_new_vectors(document_api):
    """Adding another attachment reuses existing vectors while a changed PDF receives fresh embeddings."""
    api = document_api
    requests = []
    api.controls.before_embedding = lambda: requests.append("embedding request")
    current = sync(api, add=[staged(api, "sample.py", "native_value = 3\n"),
                             staged(api, "report.pdf", pdf_bytes())])["manifest"]
    initial_calls = len(requests)
    assert initial_calls == 2
    attached = sync(api, add=[staged(api, "guide.docx", docx_bytes())])["manifest"]
    assert len(requests) == initial_calls + 1
    assert set(evidence_by_name(answer(api, uploads=context(attached)))) == {"sample.py", "report.pdf", "guide.docx"}
    pdf = next(file for file in attached["files"] if file["name"] == "report.pdf")

    changed = sync(api, add=[{**staged(api, "report.pdf", pdf_bytes(70)), "replace_file_id": pdf["file_id"]}])["manifest"]

    assert len(requests) == initial_calls + 2
    assert evidence_by_name(answer(api, uploads=context(changed)))["report.pdf"].excerpt == "Values\nPDF value | 70"
    assert list((api.root / "uploads" / "session-a" / "cache" / "embeddings").glob("*.json"))


@pytest.mark.parametrize("operation", ["clear", "exit"])
def test_clear_or_reset_releases_document_originals_and_session_caches_without_invalidating_prior_citations(document_api, operation):
    """Explicit attachment clear and session reset erase owned originals and caches while saved citations remain readable."""
    api = document_api
    current = sync(api, add=[staged(api, "report.pdf", pdf_bytes())])["manifest"]
    prior = answer(api, uploads=context(current))
    original = prior.model_dump(mode="json")
    assert owned_files(api)
    cache_root = api.root / "uploads" / "session-a" / "cache"
    assert [path for path in cache_root.rglob("*") if path.is_file()]

    cleared = (sync(api, clear=True)["manifest"] if operation == "clear" else
               final_response(api, query="exit", uploads=context(current))["upload_manifest"])

    assert cleared["files"] == []
    assert owned_files(api) == {}
    assert not [path for path in cache_root.rglob("*") if path.is_file()]
    assert prior.model_dump(mode="json") == original
    assert evidence_by_name(prior)["report.pdf"].excerpt == "Values\nPDF value | 7"


def test_document_feature_gate_rejects_pdf_without_changing_existing_native_uploads(api):
    """The default disabled document feature retains native uploads and rejects PDF attachment attempts."""
    api.settings.docling_enabled = False
    current = sync(api, add=[staged(api, "existing.py", "existing_value = 3\n")])["manifest"]

    response = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(
        current, add=[staged(api, "report.pdf", pdf_bytes())]))

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "UPLOAD_TYPE_INVALID"
    assert manifest(api) == current
    assert set(evidence_by_name(answer(api, uploads=context(current)))) == {"existing.py"}


def test_clear_after_a_failed_first_upload_erases_uncommitted_session_cache(document_api):
    """Clearing an empty manifest also releases cache created before its first candidate failed."""
    api = document_api
    initial = manifest(api)
    embedding_calls = []

    def fail_second_document():
        embedding_calls.append(True)
        if len(embedding_calls) == 2:
            raise ConnectionError("Second document embedding failed")

    api.controls.before_embedding = fail_second_document
    response = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(initial, add=[
        staged(api, "first.py", "first_value = 3\n"), staged(api, "second.pdf", pdf_bytes()),
    ]))
    assert response.status_code == 503, response.text
    assert owned_files(api) == {}
    assert manifest(api) == initial
    cache_root = api.root / "uploads" / "session-a" / "cache"
    assert [path for path in cache_root.rglob("*") if path.is_file()]

    cleared = sync(api, clear=True)["manifest"]

    assert cleared["files"] == []
    assert not [path for path in cache_root.rglob("*") if path.is_file()]


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_embedding_timeout_is_reported_as_document_timeout_without_committing_candidate_sources(document_api, cache_enabled):
    """An embedding provider timeout reports a typed document timeout and preserves the committed attachment set."""
    api = document_api
    api.settings.document_cache_enabled = cache_enabled
    current = sync(api, add=[staged(api, "report.pdf", pdf_bytes())])["manifest"]
    previous_files = owned_files(api)

    def timed_out():
        raise TimeoutError("Embedding provider request exceeded its timeout")

    api.controls.before_embedding = timed_out
    response = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(
        current, add=[staged(api, "new.docx", docx_bytes())]))

    assert response.status_code == 504, response.text
    assert response.json()["detail"]["code"] == "DOCUMENT_PROCESSING_TIMEOUT"
    assert owned_files(api) == previous_files
    assert manifest(api) == current
    assert evidence_by_name(answer(api, uploads=context(current)))["report.pdf"].excerpt == "Values\nPDF value | 7"


def test_later_embedding_batches_receive_only_the_remaining_document_deadline(document_api, monkeypatch):
    """Each document embedding request uses the budget left after earlier requests, without restarting its deadline."""
    api = document_api
    api.settings.document_upload_timeout_seconds = 10
    clock = [1000.0]
    clock_boundary = SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
    for module in ("src.app.web.upload_service", "src.infra.document_ingestion", "src.infra.tools.local_rag.uploads"):
        monkeypatch.setattr(f"{module}.time", clock_boundary)
    request_timeouts = []

    class SlowEmbeddingService(LocalEmbeddings):
        def __init__(self, *, request_timeout=None):
            super().__init__(api.controls)
            self.request_timeout = request_timeout

        def embed_documents(self, texts):
            request_timeouts.append(self.request_timeout)
            clock[0] += 4
            return super().embed_documents(texts)

    monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings",
                        lambda **kwargs: SlowEmbeddingService(request_timeout=kwargs.get("request_timeout")))

    current = sync(api, add=[staged(api, "report.pdf", pdf_bytes()),
                             staged(api, "guide.docx", docx_bytes())])["manifest"]

    assert set(evidence_by_name(answer(api, uploads=context(current)))) == {"report.pdf", "guide.docx"}
    assert request_timeouts == [10, 6]


def test_chunk_size_change_invalidates_cached_vectors_without_changing_retained_citations(document_api, monkeypatch):
    """A changed chunking configuration reembeds even unchanged short chunks while prior citations remain intact."""
    api = document_api
    embedding_requests = []
    api.controls.before_embedding = lambda: embedding_requests.append(True)
    current = sync(api, add=[staged(api, "report.pdf", pdf_bytes())])["manifest"]
    previous = answer(api, uploads=context(current))
    retained = previous.model_dump(mode="json")
    assert len(embedding_requests) == 1

    monkeypatch.setattr("src.infra.tools.local_rag.uploads.UPLOAD_CHUNK_SIZE", 400, raising=False)
    updated = sync(api, add=[staged(api, "guide.docx", docx_bytes())])["manifest"]
    latest = evidence_by_name(answer(api, uploads=context(updated)))

    assert set(latest) == {"report.pdf", "guide.docx"}
    assert latest["report.pdf"].excerpt == evidence_by_name(previous)["report.pdf"].excerpt
    assert len(embedding_requests) == 3
    assert previous.model_dump(mode="json") == retained
