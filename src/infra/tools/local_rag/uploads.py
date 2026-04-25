from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from langchain_chroma import Chroma

from src.infra.chroma_store import create_chroma_vectorstore
from src.infra.chunking import chunk_notebook_path, chunk_python_text
from src.infra.notebook_loader import ensure_canonical_upload_copy

from . import client


def _extract_text_from_py(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def extract_upload_session_id(path: str) -> str:
    parts = Path(path).expanduser().parts
    upload_index = -1
    for index, part in enumerate(parts):
        if part.lower() == "uploads":
            upload_index = index

    if upload_index < 0 or upload_index + 1 >= len(parts):
        raise ValueError("Upload path must include uploads/<session_id>/...")

    session_id = str(parts[upload_index + 1]).strip()
    if not session_id or session_id in {".", ".."}:
        raise ValueError("Upload path must include a valid session_id segment")
    return session_id


def build_upload_collection_name(session_id: str) -> str:
    normalized_session_id = re.sub(r"[^0-9A-Za-z_-]+", "-", session_id.strip()).strip("-")
    if not normalized_session_id:
        raise ValueError("session_id cannot be normalized into a collection name")
    return f"upload-session-{normalized_session_id}"


@dataclass
class UploadedRetrieverHandle:
    retriever: Any
    collection_name: str
    _vectorstore: Chroma = field(repr=False)
    _cleaned_up: bool = field(default=False, init=False, repr=False)

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._vectorstore.delete_collection()
        self._cleaned_up = True


def _chunk_upload_path(path: str) -> list[Any]:
    path_lower = str(path).lower()
    if path_lower.endswith(".py"):
        return chunk_python_text(
            path=path,
            text=_extract_text_from_py(path),
            chunk_size=800,
            chunk_overlap=120,
        )
    if path_lower.endswith(".ipynb"):
        canonical_path = ensure_canonical_upload_copy(path)
        return chunk_notebook_path(
            path=str(canonical_path),
            source_path=path,
            chunk_size=800,
            chunk_overlap=120,
        )
    raise ValueError("Unsupported file type (only .py or .ipynb).")


def build_temp_retriever(
    path: str | Sequence[str],
    api_key: str | None = None,
    k: int = 4,
) -> UploadedRetrieverHandle:
    paths = [str(path)] if isinstance(path, str) else [str(item) for item in path]
    if not paths:
        raise ValueError("At least one upload path is required.")

    session_ids = {extract_upload_session_id(item) for item in paths}
    if len(session_ids) != 1:
        raise ValueError("All upload paths must belong to the same upload session.")
    session_id = next(iter(session_ids))
    collection_name = build_upload_collection_name(session_id)

    docs: list[Any] = []
    for item in paths:
        docs.extend(_chunk_upload_path(item))

    embeddings = client.build_openai_embeddings(api_key)
    vectorstore = create_chroma_vectorstore(
        embeddings=embeddings,
        collection_name=collection_name,
    )
    if docs:
        vectorstore.add_documents(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return UploadedRetrieverHandle(
        retriever=retriever,
        collection_name=collection_name,
        _vectorstore=vectorstore,
    )
