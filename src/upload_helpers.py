from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nbformat
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .chunking import chunk_notebook, chunk_python_text


def extract_text_from_py(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file_obj:
        return file_obj.read()


def extract_text_from_ipynb(path: str) -> str:
    notebook = nbformat.read(path, as_version=4)
    texts: list[str] = []
    for cell in notebook.cells:
        if cell.get("cell_type") in {"code", "markdown"}:
            source = cell.get("source") or ""
            texts.append(str(source))
    return "\n\n".join(texts)


def extract_text(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith(".py"):
        return extract_text_from_py(path)
    if path_lower.endswith(".ipynb"):
        return extract_text_from_ipynb(path)
    raise ValueError("Unsupported file type (only .py or .ipynb).")


def _extract_upload_session_id(path: str) -> str:
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


def _build_upload_collection_name(session_id: str) -> str:
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


def build_temp_retriever(path: str, api_key: str | None = None, k: int = 4) -> UploadedRetrieverHandle:
    """Build a temporary single-session Chroma retriever from one uploaded file."""
    session_id = _extract_upload_session_id(path)
    collection_name = _build_upload_collection_name(session_id)

    path_lower = str(path).lower()
    if path_lower.endswith(".py"):
        docs = chunk_python_text(
            path=path,
            text=extract_text_from_py(path),
            chunk_size=800,
            chunk_overlap=120,
        )
    elif path_lower.endswith(".ipynb"):
        notebook = nbformat.read(path, as_version=4)
        docs = chunk_notebook(
            path=path,
            notebook=notebook,
            chunk_size=800,
            chunk_overlap=120,
        )
    else:
        raise ValueError("Unsupported file type (only .py or .ipynb).")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return UploadedRetrieverHandle(
        retriever=retriever,
        collection_name=collection_name,
        _vectorstore=vectorstore,
    )
