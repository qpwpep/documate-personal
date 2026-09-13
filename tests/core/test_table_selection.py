from __future__ import annotations

import json

import pytest

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, ParsedDocument, SourceAnchor, TableCell, TableData, build_snapshot
from src.core.evidence import build_evidence, selected_source_anchors
from src.core.planner_schema import RetrievalTask
from src.core.table_selection import table_row_units
from src.infra.chunking import chunk_parsed_document
from src.infra.tools.local_rag.serialization import build_local_hit_bundle
from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages, prepare_evidence_packet


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


def test_table_prompt_budget_preserves_target_row_numbers_and_headers():
    """A large table is reduced to a complete relevant row rather than dropped or cut into characters."""
    snapshot, element = table_source()
    item = build_evidence(snapshot=snapshot, element=element)
    wanted = build_evidence(snapshot=snapshot, element=element, cell_ids=["name", "count", "north", "alpha", "target"])
    task = RetrievalTask(route="upload", query="target", k=1, requirement={"aspects": ["target"]})
    packet = prepare_evidence_packet(
        [item], max_items=1, snippet_char_limit=len(wanted.excerpt), evidence_char_budget=len(wanted.excerpt),
        query="target", requirements_by_evidence={item.id: [task]},
    )
    assert packet == [wanted]
    assert "20" in packet[0].excerpt
    assert packet[0].element.table.cells[3].row_span == 2
    assert prepare_evidence_packet([wanted], max_items=1, snippet_char_limit=5, evidence_char_budget=5) == []


def test_table_packet_and_export_only_report_selected_cell_locations():
    """Generation and saved citations share the selected page locations and conversion limitations."""
    snapshot, element = table_source()
    page2 = SourceAnchor(kind="table", page_no=2, bbox=(10, 20, 80, 40), coordinate_space="points", precision="element")
    page3 = SourceAnchor(kind="table", page_no=3, bbox=(10, 20, 80, 40), coordinate_space="points", precision="element")
    element.anchors = [page2, page3]
    for cell in element.table.cells:
        cell.anchors = [page3 if cell.cell_id in {"south", "beta", "last"} else page2]
    item = build_evidence(snapshot=snapshot, element=element, cell_ids=["name", "count", "north", "alpha", "target"])
    assert selected_source_anchors(item) == [page2]
    messages, _, _ = build_synthesis_messages(
        state=build_graph_state_input(user_input="target"), action_rules=[], evidence_packet=[item], attempt=1, max_turns=6,
    )
    raw = str(messages[-1].content).split("\n", 2)[2]
    packet = json.loads(raw)
    assert packet[0]["source_locations"] == [page2.model_dump(mode="json")]
    assert packet[0]["quality_issues"] == snapshot.quality_issues
    assert packet[0]["is_partial"] is True
    assert {cell["cell_id"] for cell in packet[0]["table_cells"]} == set(item.selection.cell_ids)
    response = finalize_answer(text_document("target 20", basis="excerpt", refs=[item.id]), [item])
    exported = export_answer_text(response, include_sources=True)
    assert "page 2" in exported and "page 3" not in exported


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
