"""Opt-in actual Docling worker -> HTTP attachment -> Chroma -> retained citation tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from src.infra.docling_runner import DoclingRunner
from tests.web.test_multi_upload_api import api, answer, context, manifest, staged, sync  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.getenv("RUN_DOCLING_TESTS") != "1", reason="Requires prefetched local Docling/OCR models")


class TableEmbeddings(Embeddings):
    def __init__(self, calls):
        self.calls = calls

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[0.0 if " | " in text else 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [0.0, 0.0]


def test_real_pdf_docx_and_ocr_images_keep_table_pages_and_cache_ownership(api, monkeypatch):
    """Real conversion and cache reuse preserve table/page citations through HTTP replacement and deletion."""
    api.settings.docling_enabled = True
    api.settings.docling_artifacts_path = str(ROOT / "output/docling/models")
    api.settings.docling_ocr_engine = "rapidocr"
    api.settings.upload_max_file_mib = 10
    api.settings.upload_max_total_mib = 50
    calls = []
    monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: TableEmbeddings(calls))
    fixture_root = ROOT / "tests/fixtures/documents"
    names = ["bilingual.pdf", "bilingual.docx", "bilingual_scan.png", "bilingual_scan.pdf"]
    attached = sync(api, add=[staged(api, name, (fixture_root / name).read_bytes()) for name in names])["manifest"]
    original_answer = answer(api, uploads=context(attached))
    original = original_answer.model_dump(mode="json")
    by_name = {citation.evidence.snapshot.title: citation.evidence for citation in original_answer.citations}
    assert set(by_name) == set(names)
    for name, evidence in by_name.items():
        assert evidence.selection.cell_ids
        assert evidence.element.table is not None
        assert "16" in evidence.excerpt and "8" in evidence.excerpt
        if name.endswith(".docx"):
            assert not evidence.element.anchors
        else:
            assert evidence.element.anchors
            assert all(anchor.page_no >= 1 for anchor in evidence.element.anchors)
    assert by_name["bilingual_scan.png"].snapshot.quality_issues
    assert by_name["bilingual_scan.pdf"].snapshot.quality_issues
    initial_calls = len(calls)
    # A second runner whose process cannot convert anything demonstrates that
    # the on-disk cache, not a live converter object, supplies the new generation.
    cached_only = DoclingRunner(command=[sys.executable, "-c", "raise SystemExit(42)"])
    api.client.app.state.upload_service.converter = cached_only
    try:
        next_manifest = sync(api, add=[staged(api, "renamed.pdf", (fixture_root / "bilingual.pdf").read_bytes())])["manifest"]
        assert len(calls) == initial_calls
        renamed = next(file for file in next_manifest["files"] if file["name"] == "renamed.pdf")
        remaining = sync(api, remove=[file["file_id"] for file in next_manifest["files"] if file != renamed])["manifest"]
        current_answer = answer(api, uploads=context(remaining))
        assert {citation.evidence.snapshot.title for citation in current_answer.citations} == {"renamed.pdf"}
        assert {citation.evidence.element.metadata["file_id"] for citation in current_answer.citations} == {renamed["file_id"]}
        assert original_answer.model_dump(mode="json") == original
        assert not list((api.root / "uploads/session-a/conversions").glob("*/*"))
        assert cached_only.active_process_count == 0
    finally:
        cached_only.close()
