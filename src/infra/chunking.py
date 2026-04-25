from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.infra.notebook_loader import load_canonical_notebook, normalize_cell_source

_MAX_CODE_METADATA_CALLS = 8
_MAX_CODE_METADATA_KWARGS = 12
_MAX_LITERAL_CHARS = 120


def _build_splitter(*, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )


def chunk_python_text(
    *,
    path: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = _build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents([text], metadatas=[{"source": path}])
    return _annotate_python_chunks(
        path=path,
        docs=docs,
        document_char_count=len(text),
    )


def chunk_notebook_path(
    *,
    path: str,
    chunk_size: int,
    chunk_overlap: int,
    source_path: str | None = None,
) -> list[Document]:
    notebook = load_canonical_notebook(path).notebook
    return chunk_notebook(
        path=source_path or path,
        notebook=notebook,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_notebook(
    *,
    path: str,
    notebook: Any,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = _build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    base_docs: list[Document] = []
    document_char_count = 0
    for cell_index, cell in enumerate(getattr(notebook, "cells", [])):
        if cell.get("cell_type") not in {"code", "markdown"}:
            continue
        source = normalize_cell_source(cell.get("source"))
        if not source.strip():
            continue
        document_char_count += len(source)
        base_docs.append(
            Document(
                page_content=source,
                metadata={
                    "source": path,
                    "cell_id": cell_index,
                    "cell_index": cell_index,
                    "notebook_cell_id": str(cell.get("id") or "").strip() or None,
                    "cell_type": str(cell.get("cell_type") or ""),
                },
            )
        )

    if not base_docs:
        return []

    split_docs = splitter.split_documents(base_docs)
    return _annotate_notebook_chunks(
        path=path,
        docs=split_docs,
        document_char_count=document_char_count,
    )


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


def _serialize_code_metadata(*, source: str, cell_id: int | None = None) -> str | None:
    metadata = _build_code_metadata(source=source, cell_id=cell_id)
    if not metadata or not metadata.get("calls"):
        return None
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def _annotate_python_chunks(
    *,
    path: str,
    docs: list[Document],
    document_char_count: int,
) -> list[Document]:
    normalized_source = str(Path(path))
    document_chunk_count = len(docs)
    for chunk_index, doc in enumerate(docs):
        start_offset = _coerce_non_negative_int(doc.metadata.get("start_index"))
        end_offset = start_offset + len(doc.page_content or "")
        code_metadata = _serialize_code_metadata(source=doc.page_content or "")
        doc.metadata["source"] = normalized_source
        doc.metadata["chunk_id"] = chunk_index
        doc.metadata["cell_id"] = None
        doc.metadata["start_offset"] = start_offset
        doc.metadata["end_offset"] = end_offset
        doc.metadata["document_chunk_count"] = document_chunk_count
        doc.metadata["document_char_count"] = max(0, int(document_char_count))
        if code_metadata:
            doc.metadata["code_metadata"] = code_metadata
    return docs


def _annotate_notebook_chunks(
    *,
    path: str,
    docs: list[Document],
    document_char_count: int,
) -> list[Document]:
    normalized_source = str(Path(path))
    chunk_counters: dict[int, int] = {}
    document_chunk_count = len(docs)
    for doc in docs:
        cell_id = _coerce_non_negative_int(doc.metadata.get("cell_id"))
        chunk_index = chunk_counters.get(cell_id, 0)
        chunk_counters[cell_id] = chunk_index + 1
        start_offset = _coerce_non_negative_int(doc.metadata.get("start_index"))
        end_offset = start_offset + len(doc.page_content or "")
        code_metadata = None
        if str(doc.metadata.get("cell_type") or "") == "code":
            code_metadata = _serialize_code_metadata(
                source=doc.page_content or "",
                cell_id=cell_id,
            )
        doc.metadata["source"] = normalized_source
        doc.metadata["chunk_id"] = chunk_index
        doc.metadata["cell_id"] = cell_id
        doc.metadata["cell_index"] = cell_id
        doc.metadata["notebook_cell_id"] = (
            str(doc.metadata.get("notebook_cell_id") or "").strip() or None
        )
        doc.metadata["start_offset"] = start_offset
        doc.metadata["end_offset"] = end_offset
        doc.metadata["document_chunk_count"] = document_chunk_count
        doc.metadata["document_char_count"] = max(0, int(document_char_count))
        if code_metadata:
            doc.metadata["code_metadata"] = code_metadata
    return docs


def _coerce_non_negative_int(value: Any) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, coerced)
