"""Resolve explicit source ranges; prose and file names are never reparsed."""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.evidence import EvidenceRef, selected_source_anchors, source_line_records

if TYPE_CHECKING:
    from src.core.answer_schema.models import AnswerDocument, SourceLocationReference


SOURCE_LOCATION_PLACEHOLDER = "<source_location>"


def source_location_text(
    location: SourceLocationReference, evidence: EvidenceRef | None,
) -> tuple[str | None, str | None]:
    """Render only positions supported by this exact captured source selection."""
    if evidence is None or evidence.selection.cell_ids:
        return None, "source_location_unverifiable"
    selected_end = evidence.selection.end if evidence.selection.end is not None else len(evidence.element.text)
    if not evidence.selection.start <= location.start < location.end <= selected_end:
        return None, "source_location_mismatch"
    records = [row for row in source_line_records(evidence)
               if row["start"] < location.end and location.start < row["end"]]
    if not records or records[0]["start"] > location.start or records[-1]["end"] < location.end:
        return None, "source_location_unverifiable"
    title = evidence.snapshot.title or evidence.snapshot.source_uri
    cells = {(anchor.cell_id, anchor.cell_index) for anchor in selected_source_anchors(evidence)
             if anchor.kind == "notebook"}
    if len(cells) > 1:
        return None, "source_location_unverifiable"
    if cells:
        cell_id, cell_index = next(iter(cells))
        if cell_id is None and cell_index is None:
            return None, "source_location_unverifiable"
        cell = f"cell {cell_index + 1}" if cell_index is not None else f"cell ID {cell_id}"
        title += f" · {cell}"
    first, last = records[0]["line_number"], records[-1]["line_number"]
    lines = str(first) if first == last else f"{first}–{last}"
    return f"{title}:{lines}", None


def source_location_issues(document: AnswerDocument, evidence: list[EvidenceRef]) -> dict[str, list[str]]:
    """Check structured bindings and the already generated, displayed text."""
    from src.core.answer_schema.models import iter_content_units

    by_id = {item.id: item for item in evidence}
    failures: dict[str, list[str]] = {}
    for path, unit in iter_content_units(document):
        if unit.source_location is None:
            continue
        expected, issue = source_location_text(unit.source_location, by_id.get(unit.source_location.ref))
        if issue is not None:
            failures[path] = [issue]
        elif unit.text != expected:
            failures[path] = ["source_location_mismatch"]
    return failures
