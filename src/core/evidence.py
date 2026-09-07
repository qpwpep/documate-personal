"""Original-source selections and query-specific retrieval scores."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.core.documents import DocumentElement, DocumentSnapshot


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
            selected = set(self.selection.cell_ids)
            cells = sorted((cell for cell in self.element.table.cells if cell.cell_id in selected), key=lambda cell: (cell.row, cell.col))
            rows: dict[int, list[str]] = {}
            for cell in cells:
                rows.setdefault(cell.row, []).append(cell.text)
            return "\n".join(" | ".join(values) for values in rows.values())
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


def dedupe_search_hits(hits: Iterable[SearchHit]) -> list[SearchHit]:
    result: list[SearchHit] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        key = (hit.requirement_id, hit.evidence.id)
        if key not in seen:
            result.append(hit)
            seen.add(key)
    return result


def parse_search_hits(value: Any, errors: list[str] | None = None) -> list[SearchHit]:
    """Read tool hits, retaining valid items and reporting malformed entries."""
    payload = value
    if isinstance(payload, str):
        if not payload.strip():
            return []
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            if errors is not None:
                errors.append(f"search hits: invalid JSON ({exc})")
            return []
    if isinstance(payload, dict):
        payload = payload.get("hits")
    if not isinstance(payload, list):
        if errors is not None:
            errors.append("search hits: expected a list or an object containing hits")
        return []
    result = []
    for index, item in enumerate(payload):
        try:
            result.append(item if isinstance(item, SearchHit) else SearchHit.model_validate(item))
        except (ValidationError, TypeError, ValueError) as exc:
            if errors is not None:
                errors.append(f"search hits[{index}]: invalid item ({exc})")
    return dedupe_search_hits(result)
