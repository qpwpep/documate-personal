"""Original-source selections and query-specific retrieval scores."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.documents import DocumentElement, DocumentSnapshot, SourceAnchor
from src.core.python_source import PythonSourceMap
from src.core.table_selection import table_excerpt


class DocEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    type: str = ""
    default: str = ""
    description: str = ""


class DocMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_family: str = ""
    symbol: str = ""
    signature: str = ""
    parameters: list[DocEntry] = Field(default_factory=list)
    returns: list[DocEntry] = Field(default_factory=list)
    options: list[DocEntry] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    source_sections: list[str] = Field(default_factory=list)


class SourceSelection(BaseModel):
    """Half-open offsets in element.text, or structured table cell IDs."""

    model_config = ConfigDict(extra="forbid")
    start: int = Field(default=0, ge=0)
    end: int | None = Field(default=None, ge=0)
    cell_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_range(self) -> SourceSelection:
        if self.end is not None and self.end < self.start:
            raise ValueError("selection end must not precede start")
        if self.cell_ids and (self.start != 0 or self.end is not None):
            raise ValueError("cell selection cannot also select text offsets")
        if len(self.cell_ids) != len(set(self.cell_ids)):
            raise ValueError("selection cell IDs must be unique")
        return self


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    snapshot: DocumentSnapshot
    element: DocumentElement
    selection: SourceSelection = Field(default_factory=SourceSelection)

    @model_validator(mode="after")
    def validate_selection(self) -> EvidenceRef:
        selection = self.selection
        if selection.cell_ids:
            if self.element.table is None:
                raise ValueError("cell selection requires a table element")
            known = {cell.cell_id for cell in self.element.table.cells}
            if not set(selection.cell_ids).issubset(known):
                raise ValueError("selection contains an unknown table cell")
        else:
            length = len(self.element.text)
            if selection.start > length or (selection.end is not None and selection.end > length):
                raise ValueError("selection exceeds original element text")
        if self.id != _evidence_id(self.snapshot, self.element, self.selection):
            raise ValueError("evidence identity does not match its captured source and selection")
        return self

    @property
    def excerpt(self) -> str:
        if self.selection.cell_ids:
            return table_excerpt(self.element.table, self.selection.cell_ids)
        return self.element.text[self.selection.start:self.selection.end]

    @property
    def route(self) -> Literal["docs", "upload"]:
        return "docs" if self.snapshot.source_type == "official" else "upload"


class RetrievalScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str = Field(min_length=1)
    raw: float | None = Field(default=None, allow_inf_nan=False)
    direction: Literal["higher", "lower"]
    normalized: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence: EvidenceRef
    score: RetrievalScore
    rank: int = Field(ge=1)
    requirement_id: str = ""


def _evidence_id(snapshot: DocumentSnapshot, element: DocumentElement, selection: SourceSelection) -> str:
    identity = json.dumps(
        {"snapshot": snapshot.model_dump(mode="json"), "element": element.model_dump(mode="json"), "selection": selection.model_dump(mode="json")},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return "ref:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_evidence(
    *, snapshot: DocumentSnapshot, element: DocumentElement,
    start: int = 0, end: int | None = None, cell_ids: list[str] | None = None,
) -> EvidenceRef:
    """Bind a source range to its version, independently of search chunk IDs."""
    if cell_ids is None and element.table is not None and start == 0 and end is None:
        cell_ids = [cell.cell_id for cell in element.table.cells]
    cells = sorted(set(cell_ids or []))
    selection = SourceSelection(start=start, end=end, cell_ids=cells)
    if not cells:
        selection = SourceSelection(start=start, end=len(element.text) if end is None else end)
    return EvidenceRef(
        id=_evidence_id(snapshot, element, selection),
        snapshot=snapshot.model_copy(deep=True), element=element.model_copy(deep=True), selection=selection,
    )


def selected_source_anchors(evidence: EvidenceRef) -> list[SourceAnchor]:
    """Resolve known locations without inventing coordinates for cropped text.

    Missing cell provenance falls back to the complete table's locations. Exact
    text mappings refine physical lines and offsets; page boxes are never resized.
    """
    element, selection = evidence.element, evidence.selection
    if selection.cell_ids and element.table is not None:
        selected = set(selection.cell_ids)
        cells = [cell for cell in element.table.cells if cell.cell_id in selected]
        anchors = [anchor for cell in cells for anchor in cell.anchors]
        if any(not cell.anchors for cell in cells):
            anchors.extend(element.anchors)
    else:
        end = len(element.text) if selection.end is None else selection.end
        anchors = [anchor for anchor in element.anchors
                   if anchor.start is None or anchor.end is None
                   or (_text_anchor_bounds(element.text, anchor) == (0, len(element.text)))
                   or (anchor.start < end and selection.start < anchor.end)]
    unique: dict[str, SourceAnchor] = {}
    for anchor in anchors:
        if not selection.cell_ids:
            anchor = _selected_text_anchor(evidence, anchor)
        unique.setdefault(anchor.model_dump_json(), anchor)
    return [anchor.model_copy(deep=True) for anchor in unique.values()]


def _selected_text_anchor(evidence: EvidenceRef, anchor: SourceAnchor) -> SourceAnchor:
    """Narrow only an exact text mapping, never a page or approximate region."""
    bounds = _text_anchor_bounds(evidence.element.text, anchor)
    if bounds is None:
        return anchor
    source = PythonSourceMap(evidence.element.text)
    base_start, base_end = bounds
    start = max(evidence.selection.start, base_start)
    end = min(evidence.selection.end if evidence.selection.end is not None else len(source.text), base_end)
    if start >= end:
        return anchor
    offset = anchor.line_start - source.line_at_offset(base_start)
    return anchor.model_copy(update={
        "start": anchor.start + start - base_start, "end": anchor.start + end - base_start,
        "line_start": offset + source.line_at_offset(start),
        "line_end": offset + source.line_at_offset(end - 1),
    })


def _text_anchor_bounds(text: str, anchor: SourceAnchor) -> tuple[int, int] | None:
    if (anchor.kind not in {"code", "notebook", "text"} or anchor.precision != "exact"
            or anchor.start is None or anchor.end is None or anchor.line_start is None):
        return None
    if anchor.end - anchor.start == len(text) and text:
        bounds = (0, len(text))
    elif 0 <= anchor.start < anchor.end <= len(text):
        bounds = (anchor.start, anchor.end)
    else:
        return None
    source = PythonSourceMap(text)
    expected_end = anchor.line_start + source.line_at_offset(bounds[1] - 1) - source.line_at_offset(bounds[0])
    return bounds if anchor.line_end in {None, expected_end} else None


def source_line_records(evidence: EvidenceRef) -> list[dict]:
    """Number selected text using original physical lines without changing its bytes.

    Missing, partial or ambiguous exact provenance produces no invented mapping.
    Notebook numbers are cell-local, as identified by the source anchor.
    """
    if evidence.selection.cell_ids:
        return []
    selected_start = evidence.selection.start
    selected_end = evidence.selection.end if evidence.selection.end is not None else len(evidence.element.text)
    if selected_start == selected_end:
        return []
    mappings = {
        anchor.model_dump_json(): (anchor, bounds)
        for anchor in evidence.element.anchors
        if (bounds := _text_anchor_bounds(evidence.element.text, anchor)) is not None
        and bounds[0] < selected_end and selected_start < bounds[1]
    }
    if len(mappings) != 1:
        return []
    anchor, (base_start, base_end) = next(iter(mappings.values()))
    if selected_start < base_start or selected_end > base_end:
        return []
    source = PythonSourceMap(evidence.element.text)
    first = source.line_at_offset(selected_start)
    last = source.line_at_offset(selected_end - 1)
    offset = anchor.line_start - source.line_at_offset(base_start)
    records = []
    for line in range(first, last + 1):
        line_start, line_end = source.offsets[line - 1:line + 1]
        start, end = max(selected_start, line_start), min(selected_end, line_end)
        records.append({
            "line_number": offset + line,
            "start": start, "end": end, "text": source.text[start:end],
            "partial_start": start > line_start, "partial_end": end < line_end,
        })
    return records


def dedupe_search_hits(hits: Iterable[SearchHit]) -> list[SearchHit]:
    result: list[SearchHit] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        key = (hit.requirement_id, hit.evidence.id)
        if key not in seen:
            result.append(hit)
            seen.add(key)
    return result
