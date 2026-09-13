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
















