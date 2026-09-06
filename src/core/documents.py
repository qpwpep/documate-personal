"""Parser-independent document snapshots and original-source locations."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_uri: str = Field(min_length=1)
    title: str
    media_type: str
    source_type: Literal["official", "upload"]
    parser: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    parser_config: dict[str, Any] = Field(default_factory=dict)
    capture_scope: Literal["full_document", "provider_excerpt"] = "full_document"
    quality_issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity(self) -> DocumentSnapshot:
        expected_document_id = _document_id(self.source_uri, self.source_type)
        expected_snapshot_id = _snapshot_id(
            document_id=expected_document_id, content_hash=self.content_hash,
            parser=self.parser, parser_version=self.parser_version,
            parser_config=self.parser_config, capture_scope=self.capture_scope,
        )
        if self.document_id != expected_document_id or self.snapshot_id != expected_snapshot_id:
            raise ValueError("snapshot identity does not match its source and parsing revision")
        return self


class SourceAnchor(BaseModel):
    """Known source location; absent coordinates are valid, never invented.

    ``start``/``end`` are a half-open range in the adapter's original text.
    Lines/pages are one-based; notebook cell indexes are zero-based. Bounding
    boxes use ordered (left, minimum_y, right, maximum_y) coordinates.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["text", "code", "notebook", "web", "page", "table"]
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    cell_id: str | None = None
    cell_index: int | None = Field(default=None, ge=0)
    page_no: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    coordinate_space: Literal["normalized", "points", "pixels"] = "normalized"
    coordinate_origin: Literal["top_left", "bottom_left"] = "top_left"
    page_width: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    page_height: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    precision: Literal["exact", "element", "page", "document"] = "element"

    @model_validator(mode="after")
    def validate_ranges(self) -> SourceAnchor:
        for start_name, end_name in (("start", "end"), ("line_start", "line_end")):
            start, end = getattr(self, start_name), getattr(self, end_name)
            if end is not None and start is None:
                raise ValueError(f"{end_name} requires {start_name}")
            if start is not None and end is not None and end < start:
                raise ValueError(f"{end_name} must not precede {start_name}")
        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            if not all(float("-inf") < value < float("inf") for value in self.bbox):
                raise ValueError("bbox coordinates must be finite")
            if x0 > x1 or y0 > y1:
                raise ValueError("bbox coordinates must be ordered")
            if self.coordinate_space == "normalized" and not all(0 <= value <= 1 for value in self.bbox):
                raise ValueError("normalized bbox coordinates must be between 0 and 1")
            if self.page_no is None:
                raise ValueError("bbox requires page_no")
        return self


class TableCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_id: str = Field(min_length=1)
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    text: str
    is_header: bool = False
    anchors: list[SourceAnchor] = Field(default_factory=list)


class TableData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cells: list[TableCell] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cells(self) -> TableData:
        ids: set[str] = set()
        occupied: set[tuple[int, int]] = set()
        for cell in self.cells:
            if cell.cell_id in ids:
                raise ValueError(f"duplicate table cell ID: {cell.cell_id}")
            ids.add(cell.cell_id)
            positions = {(row, col) for row in range(cell.row, cell.row + cell.row_span) for col in range(cell.col, cell.col + cell.col_span)}
            if occupied.intersection(positions):
                raise ValueError("table cells must not overlap")
            occupied.update(positions)
        return self


class DocumentElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_id: str = Field(min_length=1)
    kind: Literal["heading", "paragraph", "list", "code", "table", "image"]
    text: str = ""
    parent_id: str | None = None
    order: int = Field(default=0, ge=0)
    heading_level: int | None = Field(default=None, ge=1)
    heading_path: list[str] = Field(default_factory=list)
    language: str | None = None
    table: TableData | None = None
    anchors: list[SourceAnchor] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_structure(self) -> DocumentElement:
        if self.heading_level is not None and self.kind != "heading":
            raise ValueError("heading_level belongs to heading elements")
        if self.table is not None and self.kind != "table":
            raise ValueError("table data belongs to table elements")
        if self.kind == "table" and self.table is None:
            raise ValueError("table elements require structured table data")
        return self


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot: DocumentSnapshot
    elements: list[DocumentElement]

    @model_validator(mode="after")
    def validate_hierarchy(self) -> ParsedDocument:
        by_id = {element.element_id: element for element in self.elements}
        if len(by_id) != len(self.elements):
            raise ValueError("document element IDs must be unique")
        for element in self.elements:
            seen = {element.element_id}
            parent_id = element.parent_id
            while parent_id is not None:
                if parent_id not in by_id:
                    raise ValueError(f"unknown parent element: {parent_id}")
                if parent_id in seen:
                    raise ValueError("document parent hierarchy contains a cycle")
                seen.add(parent_id)
                parent_id = by_id[parent_id].parent_id
        return self


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity_uri(source_uri: str) -> str:
    uri = source_uri.strip().replace("\\", "/")
    parsed = urlsplit(uri)
    if parsed.scheme.lower() in {"http", "https"}:
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, parsed.fragment))
    return uri


def _document_id(source_uri: str, source_type: str) -> str:
    return "doc:" + _digest(f"{source_type}:{_identity_uri(source_uri)}".encode("utf-8"))


def _snapshot_id(
    *, document_id: str, content_hash: str, parser: str, parser_version: str,
    parser_config: dict[str, Any], capture_scope: str,
) -> str:
    revision = json.dumps(
        {"document_id": document_id, "content_hash": content_hash, "parser": parser, "parser_version": parser_version, "parser_config": parser_config, "capture_scope": capture_scope},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return "snapshot:" + _digest(revision.encode("utf-8"))


def build_snapshot(
    *, source_uri: str, title: str, media_type: str,
    source_type: Literal["official", "upload"], content: bytes | str,
    parser: str, parser_version: str, parser_config: dict[str, Any] | None = None,
    capture_scope: Literal["full_document", "provider_excerpt"] = "full_document",
    quality_issues: list[str] | None = None,
) -> DocumentSnapshot:
    """Capture stable identity from original bytes and parsing configuration."""
    raw = content.encode("utf-8") if isinstance(content, str) else content
    content_hash = "sha256:" + _digest(raw)
    document_id = _document_id(source_uri, source_type)
    config = dict(parser_config or {})
    return DocumentSnapshot(
        snapshot_id=_snapshot_id(document_id=document_id, content_hash=content_hash, parser=parser, parser_version=parser_version, parser_config=config, capture_scope=capture_scope),
        document_id=document_id, content_hash=content_hash, source_uri=source_uri,
        title=title, media_type=media_type, source_type=source_type,
        parser=parser, parser_version=parser_version, parser_config=config,
        capture_scope=capture_scope, quality_issues=list(quality_issues or []),
    )
