from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.documents import (
    DocumentElement,
    ParsedDocument,
    SourceAnchor,
    TableCell,
    TableData,
    build_snapshot,
)


def snapshot(**changes):
    values = dict(
        source_uri="upload:///guide.md",
        title="guide.md",
        media_type="text/markdown",
        source_type="upload",
        content=b"# Guide\nHello",
        parser="markdown",
        parser_version="1",
    )
    values.update(changes)
    return build_snapshot(**values)


def test_snapshot_distinguishes_content_and_parser_revisions():
    """Existing citations keep their revision when source content or parsing changes."""
    initial = snapshot()
    same = snapshot()
    edited = snapshot(content=b"# Guide\nUpdated")
    reparsed = snapshot(parser_config={"headings": True})

    assert initial == same
    assert initial.document_id == edited.document_id == reparsed.document_id
    assert len({initial.snapshot_id, edited.snapshot_id, reparsed.snapshot_id}) == 3
    assert initial.content_hash == reparsed.content_hash
    assert initial.content_hash != edited.content_hash


def test_native_notebook_cell_and_original_code_layout_round_trip():
    """The document boundary preserves native cell identity, location and code text."""
    document = ParsedDocument(
        snapshot=snapshot(media_type="application/x-ipynb+json", parser="notebook"),
        elements=[
            DocumentElement(
                element_id="cell-stable-id",
                kind="code",
                text="def example():\n    return 3\n",
                order=0,
                language="python",
                heading_path=["Examples"],
                anchors=[SourceAnchor(kind="notebook", cell_id="native-abc", cell_index=2, line_start=1, line_end=2)],
            )
        ],
    )
    assert ParsedDocument.model_validate_json(document.model_dump_json()) == document


def test_document_preserves_heading_hierarchy_and_merged_table_cells():
    """Future parsers can carry headings, merged cells and page provenance losslessly."""
    document = ParsedDocument(
        snapshot=snapshot(),
        elements=[
            DocumentElement(element_id="heading", kind="heading", text="Settings", order=0, heading_level=2),
            DocumentElement(
                element_id="table", kind="table", text="Setting Value", order=1, parent_id="heading",
                table=TableData(cells=[TableCell(cell_id="header", row=0, col=0, col_span=2, text="Settings")]),
                anchors=[SourceAnchor(kind="page", page_no=3, bbox=(0.1, 0.2, 0.9, 0.5), coordinate_space="normalized", precision="element")],
            ),
        ],
    )
    assert ParsedDocument.model_validate_json(document.model_dump_json()) == document


@pytest.mark.parametrize("bbox", [(-0.1, 0.0, 0.9, 1.0), (0.2, 0.0, 1.2, 1.0), (0.9, 0.2, 0.1, 0.5)])
def test_normalized_page_regions_reject_invalid_bounds(bbox):
    """Invalid regions never reach a citation highlight renderer."""
    with pytest.raises(ValidationError):
        SourceAnchor(kind="page", page_no=1, bbox=bbox, coordinate_space="normalized")


def test_missing_page_coordinates_remain_valid():
    """A parser may identify a page without pretending to know a bounding box."""
    anchor = SourceAnchor(kind="page", page_no=2, precision="page")
    assert anchor.bbox is None


def test_document_rejects_dangling_parent():
    """A document hierarchy cannot refer to a missing element."""
    with pytest.raises(ValidationError, match="parent"):
        ParsedDocument(snapshot=snapshot(), elements=[DocumentElement(element_id="p", kind="paragraph", text="text", parent_id="missing")])


@pytest.mark.parametrize("field,value", [("source_uri", "upload:///other.md"), ("parser_version", "2"), ("parser_config", {"changed": True}), ("content_hash", "sha256:" + "0" * 64)])
def test_snapshot_rejects_identity_that_does_not_match_its_capture(field, value):
    """A stored snapshot ID cannot silently point at another source or parse revision."""
    from src.core.documents import DocumentSnapshot

    payload = snapshot().model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError, match="identity"):
        DocumentSnapshot.model_validate(payload)
