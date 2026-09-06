from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.infra.chroma_store import create_chroma_vectorstore
from src.infra.chunking import ChunkedDocument, chunk_notebook_path, chunk_python_text

from . import client


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
    _document: ChunkedDocument = field(repr=False)
    _cleaned_up: bool = field(default=False, init=False, repr=False)

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        try:
            self._vectorstore.delete_collection()
        finally:
            self._document.release()
            self._cleaned_up = True


class _SourceAwareVectorStore:
    """Hydrate only the retrieved windows, after the vector database returns them."""

    def __init__(self, vectorstore: Chroma, document: ChunkedDocument):
        self._vectorstore = vectorstore
        self._document = document

    def similarity_search_with_score(self, *args: Any, **kwargs: Any) -> list[tuple[Document, float]]:
        return [(self._document.hydrate(chunk), score)
                for chunk, score in self._vectorstore.similarity_search_with_score(*args, **kwargs)]

    def similarity_search(self, *args: Any, **kwargs: Any) -> list[Document]:
        return [self._document.hydrate(chunk) for chunk in self._vectorstore.similarity_search(*args, **kwargs)]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vectorstore, name)


class _SourceAwareRetriever:
    def __init__(self, retriever: Any, vectorstore: Chroma, document: ChunkedDocument):
        self._retriever = retriever
        self._document = document
        self.vectorstore = _SourceAwareVectorStore(vectorstore, document)

    def invoke(self, *args: Any, **kwargs: Any) -> list[Document]:
        return [self._document.hydrate(chunk) for chunk in self._retriever.invoke(*args, **kwargs)]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._retriever, name)


def build_temp_retriever(
    path: str,
    api_key: str | None = None,
    k: int = 4,
) -> UploadedRetrieverHandle:
    session_id = extract_upload_session_id(path)
    collection_name = build_upload_collection_name(session_id)

    path_lower = str(path).lower()
    if path_lower.endswith(".py"):
        source_content = Path(path).read_bytes()
        document = chunk_python_text(
            path=path,
            text=source_content.decode("utf-8"),
            chunk_size=800,
            chunk_overlap=120,
            source_content=source_content,
        )
    elif path_lower.endswith(".ipynb"):
        document = chunk_notebook_path(
            path=path,
            chunk_size=800,
            chunk_overlap=120,
        )
    else:
        raise ValueError("Unsupported file type (only .py or .ipynb).")

    embeddings = client.build_openai_embeddings(api_key)
    vectorstore = create_chroma_vectorstore(
        embeddings=embeddings,
        collection_name=collection_name,
    )
    if document.chunks:
        vectorstore.add_documents(document.chunks)
    document.chunks.clear()
    retriever = _SourceAwareRetriever(vectorstore.as_retriever(search_kwargs={"k": k}), vectorstore, document)
    return UploadedRetrieverHandle(
        retriever=retriever,
        collection_name=collection_name,
        _vectorstore=vectorstore,
        _document=document,
    )
