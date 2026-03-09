from __future__ import annotations

from pathlib import Path
from typing import Any

import nbformat
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    return _annotate_python_chunks(path=path, docs=docs)


def chunk_notebook_path(
    *,
    path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    notebook = nbformat.read(path, as_version=4)
    return chunk_notebook(
        path=path,
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
    for cell_index, cell in enumerate(getattr(notebook, "cells", [])):
        if cell.get("cell_type") not in {"code", "markdown"}:
            continue
        source = str(cell.get("source") or "")
        if not source.strip():
            continue
        base_docs.append(
            Document(
                page_content=source,
                metadata={
                    "source": path,
                    "cell_id": cell_index,
                    "cell_type": str(cell.get("cell_type") or ""),
                },
            )
        )

    if not base_docs:
        return []

    split_docs = splitter.split_documents(base_docs)
    return _annotate_notebook_chunks(path=path, docs=split_docs)


def _annotate_python_chunks(*, path: str, docs: list[Document]) -> list[Document]:
    normalized_source = str(Path(path))
    for chunk_index, doc in enumerate(docs):
        start_offset = _coerce_non_negative_int(doc.metadata.get("start_index"))
        end_offset = start_offset + len(doc.page_content or "")
        doc.metadata["source"] = normalized_source
        doc.metadata["chunk_id"] = chunk_index
        doc.metadata["cell_id"] = None
        doc.metadata["start_offset"] = start_offset
        doc.metadata["end_offset"] = end_offset
    return docs


def _annotate_notebook_chunks(*, path: str, docs: list[Document]) -> list[Document]:
    normalized_source = str(Path(path))
    chunk_counters: dict[int, int] = {}
    for doc in docs:
        cell_id = _coerce_non_negative_int(doc.metadata.get("cell_id"))
        chunk_index = chunk_counters.get(cell_id, 0)
        chunk_counters[cell_id] = chunk_index + 1
        start_offset = _coerce_non_negative_int(doc.metadata.get("start_index"))
        end_offset = start_offset + len(doc.page_content or "")
        doc.metadata["source"] = normalized_source
        doc.metadata["chunk_id"] = chunk_index
        doc.metadata["cell_id"] = cell_id
        doc.metadata["start_offset"] = start_offset
        doc.metadata["end_offset"] = end_offset
    return docs


def _coerce_non_negative_int(value: Any) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, coerced)
