"""Opt-in actual Docling worker -> HTTP attachment -> Chroma -> retained citation tests."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from src.core.answer_schema import finalize_answer, text_document
from src.core.evidence import EvidenceRef, parse_search_hits
from src.core.planner_schema import RetrievalRequirement
from src.infra.docling_runner import DoclingRunner
from src.infra.tools.local_rag import build_upload_search_tool
from src.runtime.nodes.synthesis.prompt_builder import prepare_evidence_packet
from tests.web.test_multi_upload_api import LocalChatModel, api, answer, context, manifest, staged, sync  # noqa: F401


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


class ParagraphEmbeddings(Embeddings):
    """Deterministic external embedding boundary for the fixture's distinct paragraphs."""

    terms = ("silver", "fox", "documate", "parse_document", "한글", "before", "two", "tabbed", "soft",
             "first", "second", "adjacent", "list", "marker", "field", "value", "status")

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        lowered = text.lower()
        return [float(term in lowered) for term in self.terms]


class ParagraphChatModel(LocalChatModel):
    """Replace only the external planner/synthesizer while retaining the real graph."""

    def with_structured_output(self, schema, **kwargs):
        return ParagraphChatModel(self.controls, schema_name=schema["name"])

    def invoke(self, messages):
        result = super().invoke(messages)
        if self.schema_name == "PlannerOutput":
            for task in result["parsed"]["tasks"]:
                task.update(query="quote silver fox amber trail", k=1)
        elif self.schema_name == "AnswerDocument":
            result["parsed"]["blocks"] = [
                {"type": "paragraph", "content": [block["content"]]}
                if block["type"] == "code" else block
                for block in result["parsed"]["blocks"]
            ]
        return result


@pytest.mark.parametrize("filename", ["mixed_formatting.docx", "mixed_formatting.pdf"])
def test_real_mixed_formatting_paragraphs_remain_searchable_and_exactly_citable(api, monkeypatch, filename):
    """Actual formatted paragraphs retain their boundaries and original selections through search and citations."""
    from script.generate_document_fixtures import (
        MIXED_ADJACENT, MIXED_HEADING, MIXED_INLINE, MIXED_KOREAN, MIXED_LIST,
        MIXED_SPACING, MIXED_TABLE, MIXED_TITLE, MIXED_TOKEN,
    )

    api.settings.docling_enabled = True
    api.settings.docling_artifacts_path = os.getenv("DOCLING_ARTIFACTS_PATH", str(ROOT / "output/docling/models"))
    api.settings.docling_ocr_engine = os.getenv("DOCLING_OCR_ENGINE", "rapidocr")
    api.settings.upload_max_file_mib = 10
    api.settings.upload_max_total_mib = 50
    monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: ParagraphEmbeddings())
    monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **kwargs: ParagraphChatModel(api.controls))
    raw = (ROOT / "tests/fixtures/documents" / filename).read_bytes()
    attached = sync(api, add=[staged(api, filename, raw)])["manifest"]
    file_id = attached["files"][0]["file_id"]
    handle = api.client.app.state.session_store.get_or_create("session-a").upload_retriever_handle
    parsed, = handle.retriever.source_documents
    paragraphs = [element for element in parsed.elements if element.kind == "paragraph"]
    if filename.endswith(".docx"):
        spacing_text = [MIXED_SPACING]
    else:
        # PDF stores positions rather than DOCX paragraph/tab characters. The
        # wide tab and soft break may form separate regions, but no word may be
        # lost or reordered and each returned region must retain its page box.
        korean, = [element for element in paragraphs if element.text == MIXED_KOREAN]
        adjacent, = [element for element in paragraphs if element.text == MIXED_ADJACENT[0]]
        spacing = [element for element in paragraphs if korean.order < element.order < adjacent.order]
        spacing_text = [element.text for element in spacing]
        assert re.sub(r"\s+", " ", " ".join(spacing_text)) == re.sub(r"\s+", " ", MIXED_SPACING)
        for left, right in zip(spacing, spacing[1:]):
            left_box = left.anchors[0].bbox
            right_box = right.anchors[0].bbox
            assert left_box is not None and right_box is not None
            assert left.anchors[0].coordinate_origin == right.anchors[0].coordinate_origin == "bottom_left"
            assert left_box[1] >= right_box[1]
            if abs(left_box[1] - right_box[1]) < 1:
                assert left_box[0] < right_box[0]

    expected = [MIXED_INLINE, MIXED_TOKEN, MIXED_KOREAN, *spacing_text, *MIXED_ADJACENT]
    assert [element.text for element in paragraphs] == expected
    assert [element.text for element in parsed.elements if element.kind == "list"] == [MIXED_LIST]
    for text in expected:
        element, = [element for element in paragraphs if element.text == text]
        assert MIXED_HEADING in element.heading_path
        if filename.endswith(".docx"):
            assert element.heading_path == [MIXED_TITLE, MIXED_HEADING]
            assert element.anchors == []
        else:
            assert element.anchors
            assert {anchor.page_no for anchor in element.anchors} == {1}
        # The same original paragraph must be independently searchable even when
        # font changes split words or adjacent paragraphs have identical formatting.
        payload = build_upload_search_tool()(
            query=f"quote {text}", k=1, retriever=handle.retriever,
            requirement=RetrievalRequirement(file_ids=[file_id]),
        )
        assert payload["diagnostics"]["status"] == "success", payload
        assert payload["diagnostics"]["warnings"] == []
        hit, = parse_search_hits(payload)
        assert hit.evidence.element == element
        assert hit.evidence.excerpt == text
        assert (hit.evidence.selection.start, hit.evidence.selection.end) == (0, len(text))
        assert hit.evidence.selection.cell_ids == []

    table, = [element for element in parsed.elements if element.table is not None]
    assert [[cell.text for cell in table.table.cells if cell.row == row] for row in range(3)] == MIXED_TABLE
    table_text = "\n".join(" | ".join(row) for row in MIXED_TABLE)
    table_payload = build_upload_search_tool()(
        query=f"quote {table_text}", k=1, retriever=handle.retriever,
        requirement=RetrievalRequirement(file_ids=[file_id]),
    )
    table_hit, = parse_search_hits(table_payload)
    assert table_hit.evidence.element == table
    assert table_hit.evidence.excerpt == table_text
    assert set(table_hit.evidence.selection.cell_ids) == {cell.cell_id for cell in table.table.cells}

    result = answer(api, uploads=context(attached))
    full, = [citation.evidence for citation in result.citations]
    assert full.excerpt == MIXED_INLINE
    assert full.element.text == MIXED_INLINE
    assert (full.selection.start, full.selection.end) == (0, len(MIXED_INLINE))
    assert all(check.support_status == "exact_match" for check in result.checks)
    retained = result.model_dump(mode="json")

    cropped, = prepare_evidence_packet(
        [full], max_items=1, snippet_char_limit=18, evidence_char_budget=18, query="amber trail",
    )
    assert 0 < cropped.selection.start < cropped.selection.end < len(MIXED_INLINE)
    assert cropped.excerpt == MIXED_INLINE[cropped.selection.start:cropped.selection.end]
    assert "amber trail" in cropped.excerpt
    assert cropped.element == full.element
    assert cropped.snapshot == full.snapshot
    assert cropped.id != full.id
    cropped_response = finalize_answer(
        text_document(cropped.excerpt, basis="excerpt", refs=[cropped.id]), [cropped], retrieval_required=True,
    )
    assert [check.support_status for check in cropped_response.checks] == ["exact_match"]

    sync(api, clear=True)
    assert result.model_dump(mode="json") == retained
    assert EvidenceRef.model_validate_json(cropped.model_dump_json()) == cropped
    assert cropped.element.text == MIXED_INLINE


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
