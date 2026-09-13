from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytest.importorskip("docling")

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import ConversionStatus, DoclingComponentType, ErrorItem, FailureCategory, InputFormat
from docling.datamodel.document import ConversionResult, InputDocument
from docling_core.types.doc import (
    BoundingBox, ContentLayer, CoordOrigin, DocItemLabel, DoclingDocument,
    ProvenanceItem, Size, TableCell as DoclingCell, TableData as DoclingTable,
)

from src.core.documents import ParsedDocument
from src.core.uploads import UploadRecord
from src.infra.docling_adapter import convert_file, map_docling_document
from src.infra.document_ingestion import ConversionPolicy, IngestionError


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "documents"


@pytest.fixture
def uploaded():
    path = FIXTURES / "bilingual.pdf"
    raw = path.read_bytes()
    return UploadRecord(file_id="sample", name=path.name, path=str(path), size_bytes=len(raw),
                        content_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
                        source_uri="upload://session/sample/bilingual.pdf")


def result_for(file, doc, status=ConversionStatus.SUCCESS):
    input_doc = InputDocument(path_or_stream=Path(file.path), format=InputFormat.PDF,
                              backend=DoclingParseDocumentBackend)
    return ConversionResult(input=input_doc, status=status, document=doc)


def document():
    doc = DoclingDocument(name="sample")
    doc.add_page(1, Size(width=612, height=792))
    doc.add_page(2, Size(width=612, height=792))
    return doc


def prov(page=1, box=(42, 600, 450, 650)):
    return ProvenanceItem(page_no=page,
                          bbox=BoundingBox(l=box[0], b=box[1], r=box[2], t=box[3], coord_origin=CoordOrigin.BOTTOMLEFT),
                          charspan=(0, 10))


def test_conversion_preserves_source_identity_hierarchy_and_page_coordinates(uploaded):
    """A converted passage retains its exact source bytes, heading ancestry and known page rectangle."""
    doc = document()
    title = doc.add_text(label=DocItemLabel.TITLE, text="학습 자료")
    heading = doc.add_heading("설정", level=1, parent=title)
    passage = doc.add_text(label=DocItemLabel.PARAGRAPH, text="한국어와 English", parent=heading, prov=prov(2))
    doc.add_text(label=DocItemLabel.PAGE_HEADER, text="Repeated header", content_layer=ContentLayer.FURNITURE)
    parsed = map_docling_document(result_for(uploaded, doc), uploaded, {"adapter_version": "1", "device": "cpu"})
    assert parsed.snapshot.content_hash == uploaded.content_hash
    assert parsed.snapshot.source_uri == uploaded.source_uri
    assert parsed.snapshot.title == uploaded.name
    assert parsed.snapshot.parser == "docling"
    assert parsed.snapshot.parser_config == {"adapter_version": "1", "device": "cpu"}
    assert [element.element_id for element in parsed.elements] == [title.self_ref, heading.self_ref, passage.self_ref]
    selected = parsed.elements[2]
    assert selected.parent_id == heading.self_ref
    assert selected.heading_path == ["학습 자료", "설정"]
    assert selected.text == "한국어와 English"
    assert selected.anchors[0].model_dump(exclude_none=True) == {
        "kind": "page", "page_no": 2, "bbox": (42.0, 600.0, 450.0, 650.0),
        "coordinate_space": "points", "coordinate_origin": "bottom_left",
        "page_width": 612.0, "page_height": 792.0, "precision": "element",
    }
    assert ParsedDocument.model_validate_json(parsed.model_dump_json()) == parsed


def test_flat_heading_order_builds_section_context_without_inventing_page_coordinates(uploaded):
    """Flat native office headings supply section context while unknown pages stay unknown."""
    doc = document()
    first = doc.add_heading("Overview", level=1)
    nested = doc.add_heading("Details", level=2)
    p1 = doc.add_text(label=DocItemLabel.PARAGRAPH, text="first")
    sibling = doc.add_heading("Next", level=1)
    p2 = doc.add_text(label=DocItemLabel.PARAGRAPH, text="second")
    parsed = map_docling_document(result_for(uploaded, doc), uploaded, {})
    by_id = {element.element_id: element for element in parsed.elements}
    assert by_id[p1.self_ref].heading_path == ["Overview", "Details"]
    assert by_id[p1.self_ref].parent_id == nested.self_ref
    assert by_id[nested.self_ref].parent_id == first.self_ref
    assert by_id[p2.self_ref].heading_path == ["Next"]
    assert by_id[p2.self_ref].parent_id == sibling.self_ref
    assert by_id[p2.self_ref].anchors == []


def test_table_keeps_merged_cells_all_header_roles_and_cell_positions(uploaded):
    """Table selections retain merged geometry and distinct column, row and section headers."""
    doc = document()
    cells = [
        DoclingCell(text="Group", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=2, col_span=2, column_header=True),
        DoclingCell(text="North", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=2, col_span=2, row_section=True),
        DoclingCell(text="Alpha", start_row_offset_idx=2, end_row_offset_idx=4, start_col_offset_idx=0, end_col_offset_idx=1, row_span=2, row_header=True),
        DoclingCell(text="16", start_row_offset_idx=2, end_row_offset_idx=3, start_col_offset_idx=1, end_col_offset_idx=2,
                    bbox=BoundingBox(l=100, t=100, r=180, b=120, coord_origin=CoordOrigin.TOPLEFT)),
        DoclingCell(text="8", start_row_offset_idx=3, end_row_offset_idx=4, start_col_offset_idx=1, end_col_offset_idx=2),
    ]
    table = doc.add_table(DoclingTable(table_cells=cells, num_rows=4, num_cols=2), prov=prov())
    parsed = map_docling_document(result_for(uploaded, doc), uploaded, {})
    element = parsed.elements[0]
    assert element.element_id == table.self_ref
    assert [(cell.row, cell.col, cell.row_span, cell.col_span, cell.text) for cell in element.table.cells] == [
        (0, 0, 1, 2, "Group"), (1, 0, 1, 2, "North"), (2, 0, 2, 1, "Alpha"), (2, 1, 1, 1, "16"), (3, 1, 1, 1, "8"),
    ]
    ids = [cell.cell_id for cell in element.table.cells]
    assert element.metadata["table_header_cell_ids"] == {"column": [ids[0]], "row": [ids[2]], "section": [ids[1]]}
    assert element.table.cells[3].anchors[0].bbox == (100, 100, 180, 120)
    assert element.table.cells[3].anchors[0].coordinate_origin == "top_left"
    assert element.table.cells[4].anchors[0].precision == "page"
    assert element.table.cells[4].anchors[0].bbox is None


@pytest.mark.parametrize("status", [ConversionStatus.PARTIAL_SUCCESS, ConversionStatus.FAILURE, ConversionStatus.SKIPPED])
def test_non_successful_results_never_become_searchable_documents(uploaded, status):
    """Partial and failed conversions cannot silently publish their surviving paragraphs."""
    doc = document()
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="surviving text")
    with pytest.raises(IngestionError) as caught:
        map_docling_document(result_for(uploaded, doc, status), uploaded, {})
    expected = "DOCUMENT_PARTIAL_CONVERSION" if status == ConversionStatus.PARTIAL_SUCCESS else "DOCUMENT_INVALID"
    assert caught.value.code == expected
    assert caught.value.file_name == uploaded.name


def test_success_status_with_missing_source_pages_is_rejected(uploaded):
    """A backend success flag cannot hide a missing source page from a full-document snapshot."""
    doc = document()
    del doc.pages[2]
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="page one")
    with pytest.raises(IngestionError) as caught:
        map_docling_document(result_for(uploaded, doc), uploaded, {})
    assert caught.value.code == "DOCUMENT_PARTIAL_CONVERSION"


def test_picture_only_document_reports_no_searchable_content(uploaded):
    """An image placeholder is insufficient evidence that searchable text was extracted."""
    doc = document()
    doc.add_picture(prov=prov())
    with pytest.raises(IngestionError) as caught:
        map_docling_document(result_for(uploaded, doc), uploaded, {})
    assert caught.value.code == "DOCUMENT_NO_SEARCHABLE_CONTENT"


def test_source_change_during_conversion_invalidates_the_result(uploaded, tmp_path):
    """A source replacement cannot attach converted content to another byte revision."""
    path = tmp_path / "changed.pdf"
    path.write_bytes(Path(uploaded.path).read_bytes())
    copy = uploaded.model_copy(update={"path": str(path)})
    doc = document()
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="source")
    result = result_for(copy, doc)
    path.write_bytes(b"replaced source")
    with pytest.raises(IngestionError) as caught:
        map_docling_document(result, copy, {})
    assert caught.value.code == "DOCUMENT_SOURCE_CHANGED"


def test_timeout_is_reported_even_if_docling_returns_partial_text(uploaded):
    """Docling's soft timeout remains a distinct failure instead of publishing partial text."""
    doc = document()
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="before timeout")
    result = result_for(uploaded, doc, ConversionStatus.PARTIAL_SUCCESS)
    result.errors = [ErrorItem(component_type=DoclingComponentType.PIPELINE, module_name="pipeline",
                               error_message="timeout", category=FailureCategory.TIMEOUT)]
    with pytest.raises(IngestionError) as caught:
        map_docling_document(result, uploaded, {})
    assert caught.value.code == "DOCUMENT_PROCESSING_TIMEOUT"
    assert caught.value.retryable


def test_multi_page_table_never_guesses_a_cells_page_from_its_rectangle(uploaded):
    """Cell rectangles without page IDs stay ambiguous when their table spans pages."""
    doc = document()
    cell = DoclingCell(text="value", start_row_offset_idx=0, end_row_offset_idx=1,
                       start_col_offset_idx=0, end_col_offset_idx=1, bbox=prov().bbox)
    item = doc.add_table(DoclingTable(table_cells=[cell], num_rows=1, num_cols=1), prov=prov(1))
    item.prov.append(prov(2))
    parsed = map_docling_document(result_for(uploaded, doc), uploaded, {})
    assert parsed.elements[0].table.cells[0].anchors == []
    assert [anchor.page_no for anchor in parsed.elements[0].anchors] == [1, 2]


@pytest.mark.parametrize("ocr_enabled, warning", [
    (True, "OCR을 포함한 자동 변환 결과에는 글자 인식 오류나 내용 누락이 있을 수 있습니다."),
    (False, "OCR을 사용하지 않아 이미지 안의 문자는 검색 대상에 포함되지 않을 수 있습니다."),
])
def test_pdf_snapshot_reports_the_actual_ocr_policy_limitations(uploaded, ocr_enabled, warning):
    """Source citations retain the known OCR policy limitation with their converted text."""
    doc = document()
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="text")
    parsed = map_docling_document(result_for(uploaded, doc), uploaded, {"do_ocr": ocr_enabled})
    assert parsed.snapshot.quality_issues == [warning]


def test_pdf_over_page_limit_reports_a_limit_without_loading_models(uploaded, tmp_path):
    """A valid oversized PDF is distinguished from a broken file before model loading."""
    with pytest.raises(IngestionError) as caught:
        convert_file(uploaded, ConversionPolicy(artifacts_path=str(tmp_path), max_pdf_pages=1))
    assert caught.value.code == "DOCUMENT_LIMIT_EXCEEDED"


@pytest.mark.skipif(os.getenv("RUN_DOCLING_TESTS") != "1", reason="RUN_DOCLING_TESTS=1 requires prefetched local models")
@pytest.mark.parametrize("filename", ["bilingual.pdf", "bilingual.docx", "bilingual_scan.pdf", "bilingual_scan.png"])
def test_local_conversion_recovers_fixture_facts_and_structured_table(filename):
    """Actual local conversion recovers bilingual facts, table cells and honest source locations."""
    path = FIXTURES / filename
    raw = path.read_bytes()
    file = UploadRecord(file_id=filename, name=filename, path=str(path), size_bytes=len(raw),
                        content_hash="sha256:" + hashlib.sha256(raw).hexdigest(), source_uri=f"upload://live/{filename}")
    model_path = os.getenv("DOCLING_ARTIFACTS_PATH", str(FIXTURES.parents[2] / "output/docling/models"))
    policy = ConversionPolicy(artifacts_path=model_path, ocr_engine=os.getenv("DOCLING_OCR_ENGINE", "rapidocr"))
    parsed = convert_file(file, policy)
    text = " ".join(element.text for element in parsed.elements)
    assert "DOCUMATE-2026" in text
    tables = [element for element in parsed.elements if element.table]
    assert tables
    assert {"16", "8"}.issubset({cell.text for table in tables for cell in table.table.cells})
    assert parsed.snapshot.content_hash == file.content_hash
    if filename.endswith(".docx"):
        assert all(not element.anchors for element in parsed.elements)
    else:
        assert any(anchor.page_no == 1 for element in parsed.elements for anchor in element.anchors)
