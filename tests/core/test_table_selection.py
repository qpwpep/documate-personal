from __future__ import annotations


import pytest

from src.core.documents import DocumentElement, ParsedDocument, SourceAnchor, TableCell, TableData, build_snapshot
from src.core.evidence import build_evidence, selected_source_anchors
from src.core.table_selection import table_row_units
from src.infra.chunking import chunk_parsed_document
from src.infra.tools.local_rag.serialization import build_local_hit_bundle


def table_source():
    snapshot = build_snapshot(
        source_uri="upload:///counts.pdf", title="counts.pdf", media_type="application/pdf",
        source_type="upload", content=b"original pdf", parser="docling", parser_version="test",
        quality_issues=["그림의 내용은 추출하지 않았습니다."],
    )
    element = DocumentElement(element_id="counts", kind="table", table=TableData(cells=[
        TableCell(cell_id="name", row=0, col=0, text="Name", is_header=True),
        TableCell(cell_id="count", row=0, col=1, text="Count", is_header=True),
        TableCell(cell_id="north", row=1, col=0, col_span=2, text="North", is_header=True),
        TableCell(cell_id="alpha", row=2, col=0, row_span=2, text="Alpha", is_header=True),
        TableCell(cell_id="first", row=2, col=1, text="10"),
        TableCell(cell_id="target", row=3, col=1, text="target 20"),
        TableCell(cell_id="south", row=4, col=0, col_span=2, text="South", is_header=True),
        TableCell(cell_id="beta", row=5, col=0, text="Beta", is_header=True),
        TableCell(cell_id="last", row=5, col=1, text="30"),
    ]), metadata={"table_header_cell_ids": {
        "column": ["name", "count"], "row": ["alpha", "beta"], "section": ["north", "south"],
    }})
    return snapshot, element


def test_table_rows_keep_their_own_section_and_merged_header():
    """A selected data row retains its headers without pulling in another section or data row."""
    _, element = table_source()
    units = [set(ids) for ids in table_row_units(element)]
    assert units == [
        {"name", "count", "north", "alpha", "first"},
        {"name", "count", "north", "alpha", "target"},
        {"name", "count", "south", "beta", "last"},
    ]


def test_table_row_selection_never_expands_an_allowed_source_selection():
    """Unavailable headers and cells are not read back from the retained full table."""
    _, element = table_source()
    allowed = ["name", "count", "north", "alpha", "target"]
    assert [set(ids) for ids in table_row_units(element, allowed_cell_ids=allowed)] == [set(allowed)]
    assert table_row_units(element, allowed_cell_ids=["alpha", "target"]) == []


def test_structured_table_indexes_hydrates_and_survives_source_release():
    """An empty text body still produces searchable, self-contained cell citations."""
    snapshot, element = table_source()
    indexed = chunk_parsed_document(ParsedDocument(snapshot=snapshot, elements=[element]), chunk_size=45, chunk_overlap=10)
    assert indexed.chunks
    assert all(chunk.metadata["selection_kind"] == "table_cells" for chunk in indexed.chunks)
    assert all(isinstance(chunk.metadata["cell_ids_json"], str) for chunk in indexed.chunks)
    hydrated = [indexed.hydrate(chunk) for chunk in indexed.chunks]
    hits, _, _, warnings = build_local_hit_bundle([(chunk, 0.2) for chunk in hydrated], query="target")
    assert not warnings
    assert all(hit.evidence.selection.cell_ids for hit in hits)
    assert [hit.evidence.excerpt for hit in hits] == [chunk.page_content for chunk in hydrated]
    before = [hit.model_dump(mode="json") for hit in hits]
    indexed.release()
    assert [hit.model_dump(mode="json") for hit in hits] == before


@pytest.mark.parametrize("corruption", ["not json", '["unknown"]', '[]', '{"cell":"first"}'])
def test_bad_table_chunk_selection_is_rejected(corruption):
    """A corrupt cell reference cannot fall back to an unrelated text or whole-table selection."""
    snapshot, element = table_source()
    indexed = chunk_parsed_document(ParsedDocument(snapshot=snapshot, elements=[element]), chunk_size=45, chunk_overlap=10)
    chunk = indexed.chunks[0].model_copy(deep=True)
    chunk.metadata["cell_ids_json"] = corruption
    with pytest.raises(ValueError):
        indexed.hydrate(chunk)






def test_text_page_locations_follow_selected_charspan_without_resizing_bbox():
    """Cropping an extracted passage filters pages but never invents a smaller physical region."""
    snapshot, _ = table_source()
    anchors = [SourceAnchor(kind="page", start=0, end=5, page_no=1, precision="element"),
               SourceAnchor(kind="page", start=5, end=10, page_no=2, bbox=(0.1, 0.2, 0.5, 0.9), precision="element")]
    element = DocumentElement(element_id="p", kind="paragraph", text="firstlater", anchors=anchors)
    item = build_evidence(snapshot=snapshot, element=element, start=6, end=9)
    assert selected_source_anchors(item) == [anchors[1]]
    assert item.element.anchors == anchors


def test_table_without_cell_page_provenance_keeps_all_known_table_pages():
    """A multi-page table never assigns an unknown cell location to its first page."""
    snapshot, element = table_source()
    element.anchors = [SourceAnchor(kind="page", page_no=1, precision="element"),
                       SourceAnchor(kind="page", page_no=2, precision="element")]
    item = build_evidence(snapshot=snapshot, element=element, cell_ids=["target"])
    assert selected_source_anchors(item) == element.anchors
