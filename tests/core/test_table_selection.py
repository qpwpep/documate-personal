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
from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
from src.runtime.nodes.synthesis.evidence_selection import select_evidence_packet


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




