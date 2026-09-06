from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.documents import DocumentElement, ParsedDocument, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence
from src.infra.notebook_loader import load_canonical_notebook, normalize_cell_source

_MAX_CODE_METADATA_CALLS = 8
_MAX_CODE_METADATA_KWARGS = 12
_MAX_LITERAL_CHARS = 120


@dataclass
class ChunkedDocument:
    """One source registry and lightweight text windows ready for indexing."""

    parsed: ParsedDocument
    chunks: list[Document]
    _elements: dict[str, DocumentElement] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._elements = {element.element_id: element for element in self.parsed.elements}

    def hydrate(self, chunk: Document) -> Document:
        metadata = chunk.metadata
        if metadata.get("snapshot_id") != self.parsed.snapshot.snapshot_id:
            raise ValueError("Indexed chunk belongs to a different source revision")
        element = self._elements.get(str(metadata.get("element_id") or ""))
        if element is None:
            raise ValueError("Indexed chunk points to an unknown source element")
        evidence = build_evidence(
            snapshot=self.parsed.snapshot, element=element,
            start=int(metadata["start"]), end=int(metadata["end"]),
        )
        if evidence.excerpt != chunk.page_content:
            raise ValueError("Indexed text differs from its source range")
        return Document(page_content=chunk.page_content,
                        metadata={**metadata, "evidence_ref": evidence.model_dump_json()})

    def release(self) -> None:
        """Release index-owned source data; previously returned evidence is independent."""
        self._elements.clear()
        self.parsed.elements.clear()
        self.chunks.clear()


def _build_splitter(*, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=False,
    )


def chunk_python_text(
    *,
    path: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    source_content: bytes | None = None,
) -> ChunkedDocument:
    snapshot = build_snapshot(
        source_uri=path, title=Path(path).name, media_type="text/x-python",
        source_type="upload", content=source_content if source_content is not None else text,
        parser="python-ast", parser_version="1",
    )
    element = DocumentElement(
        element_id="python-source", kind="code", text=text, language="python",
        anchors=[SourceAnchor(kind="code", start=0, end=len(text), line_start=1,
                              line_end=max(1, len(text.splitlines())), precision="exact")],
        metadata={"code_metadata": _build_code_metadata(source=text)},
    )
    return chunk_parsed_document(ParsedDocument(snapshot=snapshot, elements=[element]),
                                 chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_notebook_path(
    *,
    path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> ChunkedDocument:
    loaded = load_canonical_notebook(path)
    return chunk_notebook(
        path=path,
        notebook=loaded.notebook,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_content=loaded.source_content,
    )


def chunk_notebook(
    *,
    path: str,
    notebook: Any,
    chunk_size: int,
    chunk_overlap: int,
    source_content: bytes | str | None = None,
) -> ChunkedDocument:
    snapshot = build_snapshot(
        source_uri=path, title=Path(path).name, media_type="application/x-ipynb+json",
        source_type="upload", content=source_content if source_content is not None else json.dumps(notebook, ensure_ascii=False),
        parser="notebook-source", parser_version="1", parser_config={"line_endings": "LF"},
    )
    elements: list[DocumentElement] = []
    for cell_index, cell in enumerate(getattr(notebook, "cells", [])):
        if cell.get("cell_type") not in {"code", "markdown"}:
            continue
        source = normalize_cell_source(cell.get("source"))
        if not source.strip():
            continue
        native_id = str(cell.get("id") or "").strip() or f"cell-{cell_index}"
        is_code = cell.get("cell_type") == "code"
        elements.append(DocumentElement(
            element_id=f"cell-{native_id}", kind="code" if is_code else "paragraph",
            text=source, order=cell_index, language="python" if is_code else None,
            anchors=[SourceAnchor(kind="notebook", start=0, end=len(source),
                                  line_start=1, line_end=max(1, len(source.splitlines())),
                                  cell_id=native_id, cell_index=cell_index, precision="exact")],
            metadata={"cell_type": str(cell.get("cell_type")),
                      "code_metadata": _build_code_metadata(source=source, cell_id=cell_index) if is_code else {}},
        ))
    return chunk_parsed_document(ParsedDocument(snapshot=snapshot, elements=elements),
                                 chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = [node.attr]
        value = node.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def _literal_text(node: ast.AST) -> str:
    try:
        rendered = ast.unparse(node).strip()
    except Exception:
        return ""
    if len(rendered) <= _MAX_LITERAL_CHARS:
        return rendered
    return rendered[: _MAX_LITERAL_CHARS - 3].rstrip() + "..."


def _option_literal(name: str, value: str) -> str:
    return f"{name}={value}"


def _build_code_metadata(*, source: str, cell_id: int | None = None) -> dict[str, Any]:
    if not str(source or "").strip():
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    calls: list[dict[str, Any]] = []
    option_literals: list[str] = []
    seen_options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if not call_name:
            continue

        kwargs: dict[str, str] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            value = _literal_text(keyword.value)
            if not value:
                continue
            kwargs[keyword.arg] = value
            option = _option_literal(keyword.arg, value)
            compact = "".join(option.lower().split())
            if compact not in seen_options:
                option_literals.append(option)
                seen_options.add(compact)
            if len(kwargs) >= _MAX_CODE_METADATA_KWARGS:
                break

        call_payload: dict[str, Any] = {"call_name": call_name}
        if kwargs:
            call_payload["kwargs"] = kwargs
        line_number = getattr(node, "lineno", None)
        if isinstance(line_number, int) and line_number > 0:
            call_payload["line"] = line_number
        calls.append(call_payload)
        if len(calls) >= _MAX_CODE_METADATA_CALLS:
            break

    metadata: dict[str, Any] = {}
    if cell_id is not None:
        metadata["cell_id"] = max(0, int(cell_id))
    if calls:
        metadata["calls"] = calls
    if option_literals:
        metadata["option_literals"] = option_literals[:_MAX_CODE_METADATA_KWARGS]
    return metadata


def chunk_parsed_document(
    parsed: ParsedDocument, *, chunk_size: int, chunk_overlap: int,
) -> ChunkedDocument:
    """Store each source element once and index only its stable location."""
    splitter = _build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: list[Document] = []
    document_char_count = sum(len(element.text) for element in parsed.elements)
    for element in parsed.elements:
        for chunk in splitter.create_documents([element.text]):
            start = int(chunk.metadata.get("start_index", -1))
            end = start + len(chunk.page_content)
            if start < 0 or element.text[start:end] != chunk.page_content:
                raise ValueError("Chunk text does not resolve to its source element")
            metadata: dict[str, Any] = {
                "source": parsed.snapshot.source_uri,
                "snapshot_id": parsed.snapshot.snapshot_id,
                "element_id": element.element_id,
                "start": start,
                "end": end,
                "document_char_count": document_char_count,
            }
            anchor = next((anchor for anchor in element.anchors if anchor.kind == "notebook"), None)
            if anchor is not None:
                metadata["cell_index"] = anchor.cell_index
            docs.append(Document(page_content=chunk.page_content, metadata=metadata))
    for doc in docs:
        doc.metadata["document_chunk_count"] = len(docs)
    return ChunkedDocument(parsed=parsed.model_copy(deep=True), chunks=docs)
