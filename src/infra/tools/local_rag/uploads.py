from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from importlib.metadata import version
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from nbformat import from_dict
from openai import APITimeoutError

from src.core.documents import ParsedDocument
from src.core.uploads import UploadFileInfo, UploadRecord
from src.core.upload_formats import is_document_upload
from src.infra.chroma_store import create_chroma_vectorstore, add_preembedded_documents
from src.infra.chunking import ChunkedDocument, chunk_notebook, chunk_notebook_path, chunk_python_text, chunk_parsed_document
from src.infra.document_ingestion import DocumentIngestionContext, IngestionError
from src.infra.embedding_cache import CachedEmbeddings
from src.infra.notebook_loader import canonicalize_notebook_payload

from . import client


UPLOAD_CHUNK_SIZE = 800
UPLOAD_CHUNK_OVERLAP = 120
UPLOAD_CHUNKING_VERSION = "source-ranges-and-table-cells-v1"


@dataclass(frozen=True)
class _PendingIndexDeletion:
    client: Any = field(repr=False)
    name: str
    collection_id: str


_pending_index_deletions: OrderedDict[tuple[int, str], _PendingIndexDeletion] = OrderedDict()
_index_cleanup_lock = Lock()
_INDEX_CLEANUP_BATCH = 16


def retry_pending_upload_index_cleanup() -> None:
    """Retry a bounded batch without retaining retrievers or released source documents.

    New managed builds stop while this backlog remains. Failures can therefore
    add only already-live/in-flight generations, rather than an unlimited series
    of new candidates. Legacy collections never enter this registry.
    """
    with _index_cleanup_lock:
        pending = list(_pending_index_deletions.items())[:_INDEX_CLEANUP_BATCH]
    for key, item in pending:
        try:
            try:
                current = item.client.get_collection(item.name)
            except NotFoundError:
                current = None
            # A name reused outside the managed builder is not the retired resource.
            if current is not None and str(current.id) == item.collection_id:
                item.client.delete_collection(item.name)
        except Exception:
            logging.getLogger(__name__).warning("upload_index_cleanup_retry_failed", exc_info=True)
            with _index_cleanup_lock:
                if key in _pending_index_deletions:
                    _pending_index_deletions.move_to_end(key)
        else:
            with _index_cleanup_lock:
                if _pending_index_deletions.get(key) is item:
                    del _pending_index_deletions[key]


def _delete_upload_index(vectorstore: Chroma, *, managed: bool) -> None:
    collection = vectorstore._collection
    key = (id(vectorstore._client), str(collection.id))
    try:
        current = vectorstore._client.get_collection(collection.name) if managed else collection
        if str(current.id) == str(collection.id):
            vectorstore.delete_collection()
    except NotFoundError:
        pass
    except Exception:
        if managed:
            with _index_cleanup_lock:
                _pending_index_deletions[key] = _PendingIndexDeletion(
                    client=vectorstore._client, name=collection.name, collection_id=str(collection.id))
        raise
    with _index_cleanup_lock:
        _pending_index_deletions.pop(key, None)


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
    _documents: tuple[ChunkedDocument, ...] = field(repr=False)
    _managed: bool = field(default=False, repr=False)
    _cleaned_up: bool = field(default=False, init=False, repr=False)

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        try:
            _delete_upload_index(self._vectorstore, managed=self._managed)
            self._cleaned_up = True
        finally:
            for document in self._documents:
                document.release()
            # Legacy names can be reused even after failed deletion. Preserve
            # their one-shot disposal; only isolated managed generations retry.
            if not self._managed:
                self._cleaned_up = True


class _SourceRegistry:
    def __init__(self, documents: tuple[ChunkedDocument, ...]):
        self.documents = documents
        self.by_snapshot = {document.parsed.snapshot.snapshot_id: document for document in documents}

    def hydrate(self, chunk: Document) -> Document:
        document = self.by_snapshot.get(str(chunk.metadata.get("snapshot_id") or ""))
        if document is None:
            raise ValueError("Indexed chunk belongs to an unknown source revision")
        return document.hydrate(chunk)


class _SourceAwareVectorStore:
    """Hydrate only the retrieved windows, after the vector database returns them."""

    def __init__(self, vectorstore: Chroma, registry: _SourceRegistry, file_ids: tuple[str, ...] = ()):
        self._vectorstore = vectorstore
        self._registry = registry
        self._file_ids = file_ids

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None, **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        file_ids = tuple(dict.fromkeys(kwargs.pop("file_ids", ()) or ()))
        if set(file_ids).difference(self._file_ids):
            raise ValueError("Upload file scope is not active in this session")
        if not file_ids:
            rows = self._vectorstore.similarity_search_with_score(
                query, k=k, filter=filter, where_document=where_document, **kwargs)
        else:
            # Reserve at least one candidate per explicitly requested source.
            budget = max(int(k), len(file_ids))
            embeddings = self._vectorstore.embeddings
            if embeddings is None:
                raise ValueError("Scoped upload retrieval requires a query embedding function")
            query_embedding = embeddings.embed_query(query)
            groups = []
            for file_id in file_ids:
                scope = {"file_id": file_id}
                if filter:
                    scope = {"$and": [scope, filter]}
                # This vector API returns raw distances, just like the text API.
                groups.append(self._vectorstore.similarity_search_by_vector_with_relevance_scores(
                    query_embedding, k=budget, filter=scope, where_document=where_document, **kwargs))
            rows = [group[0] for group in groups if group]
            remaining = sorted((item for group in groups for item in group[1:]), key=lambda item: item[1])
            rows.extend(remaining[:max(0, budget - len(rows))])
        return [(self._registry.hydrate(chunk), score) for chunk, score in rows]

    def similarity_search(self, *args: Any, **kwargs: Any) -> list[Document]:
        return [chunk for chunk, _score in self.similarity_search_with_score(*args, **kwargs)]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vectorstore, name)


class _SourceAwareRetriever:
    def __init__(self, retriever: Any, vectorstore: Chroma, documents: tuple[ChunkedDocument, ...],
                 upload_files: tuple[UploadFileInfo, ...] = ()):
        self._retriever = retriever
        self._registry = _SourceRegistry(documents)
        self.upload_files = upload_files
        self.vectorstore = _SourceAwareVectorStore(vectorstore, self._registry,
                                                  tuple(file.file_id for file in upload_files))

    @property
    def source_documents(self) -> tuple[ParsedDocument, ...]:
        return tuple(document.parsed for document in self._registry.documents if document.parsed.elements)

    @property
    def source_document(self) -> ParsedDocument | None:
        """The current session's preserved source; no filesystem or cross-session lookup."""
        documents = self.source_documents
        return documents[0] if len(documents) == 1 else None

    def invoke(self, *args: Any, **kwargs: Any) -> list[Document]:
        return [self._registry.hydrate(chunk) for chunk in self._retriever.invoke(*args, **kwargs)]

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
            chunk_size=UPLOAD_CHUNK_SIZE,
            chunk_overlap=UPLOAD_CHUNK_OVERLAP,
            source_content=source_content,
        )
    elif path_lower.endswith(".ipynb"):
        document = chunk_notebook_path(
            path=path,
            chunk_size=UPLOAD_CHUNK_SIZE,
            chunk_overlap=UPLOAD_CHUNK_OVERLAP,
        )
    else:
        raise ValueError("Unsupported file type (only .py or .ipynb).")

    return _index_documents((document,), collection_name=collection_name, api_key=api_key, k=k)


def _index_documents(
    documents: tuple[ChunkedDocument, ...], *, collection_name: str,
    api_key: str | None, k: int, upload_files: tuple[UploadFileInfo, ...] = (),
    ingestion: DocumentIngestionContext | None = None,
) -> UploadedRetrieverHandle:
    vectorstore = None
    try:
        try:
            if ingestion is not None:
                ingestion.check_deadline()
            embeddings = client.build_openai_embeddings(api_key)
            vectorstore = create_chroma_vectorstore(
                embeddings=embeddings, collection_name=collection_name)
        except IngestionError:
            raise
        except Exception as exc:
            raise RuntimeError("Upload index initialization failed") from exc
        # Keep each embedding/storage request bounded even for a large notebook.
        for document in documents:
            try:
                for start in range(0, len(document.chunks), 128):
                    if ingestion is not None:
                        ingestion.check_deadline()
                    batch = document.chunks[start:start + 128]
                    if ingestion is not None:
                        provider = embeddings
                        if ingestion.deadline is not None and hasattr(embeddings, "request_timeout"):
                            provider = client.build_openai_embeddings(api_key, timeout_seconds=max(0.1, ingestion.deadline - time.monotonic()))
                        if ingestion.cache_dir is not None:
                            snapshot = document.parsed.snapshot
                            provider = CachedEmbeddings(provider, ingestion.cache_dir / "embeddings", namespace={
                                "source_hash": snapshot.content_hash, "parser": snapshot.parser,
                                "parser_version": snapshot.parser_version, "parser_config": snapshot.parser_config,
                                "chunking": {"version": UPLOAD_CHUNKING_VERSION, "size": UPLOAD_CHUNK_SIZE, "overlap": UPLOAD_CHUNK_OVERLAP},
                                "embedding": {"model": getattr(embeddings, "model", "test-embedding-boundary"),
                                    "dimensions": getattr(embeddings, "dimensions", None),
                                    "api_base": getattr(embeddings, "openai_api_base", None),
                                    "sdk_version": version("langchain-openai")},
                            }, max_bytes=ingestion.cache_max_bytes, ttl_seconds=ingestion.cache_ttl_seconds)
                        vectors = provider.embed_documents([chunk.page_content for chunk in batch])
                        ingestion.check_deadline()
                        add_preembedded_documents(vectorstore, batch, vectors)
                    else:
                        vectorstore.add_documents(batch)
                if ingestion is not None:
                    ingestion.check_deadline()
            except IngestionError:
                raise
            except Exception as exc:
                if ingestion is not None:
                    ingestion.check_deadline()
                    if isinstance(exc, (TimeoutError, APITimeoutError)):
                        raise IngestionError("DOCUMENT_PROCESSING_TIMEOUT", "문서 임베딩 요청 시간이 초과되었습니다.",
                                             file_name=document.parsed.snapshot.title, retryable=True) from exc
                raise RuntimeError(f"{document.parsed.snapshot.title}: embedding or vector insertion failed") from exc
            document.chunks.clear()
        retriever = _SourceAwareRetriever(vectorstore.as_retriever(search_kwargs={"k": k}),
                                          vectorstore, documents, upload_files)
        return UploadedRetrieverHandle(retriever=retriever, collection_name=collection_name,
                                       _vectorstore=vectorstore, _documents=documents, _managed=bool(upload_files))
    except BaseException:
        if vectorstore is not None:
            try:
                _delete_upload_index(vectorstore, managed=bool(upload_files))
            except Exception:
                logging.getLogger(__name__).exception("Failed to clean up candidate upload index")
        for document in documents:
            document.release()
        raise


def _parse_upload(file: UploadRecord) -> ChunkedDocument:
    content = Path(file.path).read_bytes()
    if len(content) != file.size_bytes or "sha256:" + hashlib.sha256(content).hexdigest() != file.content_hash:
        raise ValueError("uploaded file changed after validation")
    options = {"path": file.path, "source_uri": file.source_uri, "title": file.name,
               "source_content": content, "chunk_size": UPLOAD_CHUNK_SIZE, "chunk_overlap": UPLOAD_CHUNK_OVERLAP}
    if file.name.lower().endswith(".py"):
        document = chunk_python_text(text=content.decode("utf-8"), **options)
    elif file.name.lower().endswith(".ipynb"):
        payload = json.loads(content.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("notebook payload must be an object")
        canonical, _added = canonicalize_notebook_payload(payload)
        document = chunk_notebook(notebook=from_dict(canonical), **options)
    else:
        raise ValueError("Unsupported file type (only .py or .ipynb).")
    if not document.chunks:
        document.release()
        raise ValueError("file has no searchable text")
    for element in document.parsed.elements:
        element.metadata["file_id"] = file.file_id
    for chunk in document.chunks:
        chunk.metadata["file_id"] = file.file_id
    return document


def build_upload_retriever(
    files: list[UploadRecord], *, session_id: str, generation: str,
    api_key: str | None = None, k: int = 4, ingestion: DocumentIngestionContext | None = None,
) -> UploadedRetrieverHandle:
    """Build an isolated candidate index without touching the currently active generation."""
    retry_pending_upload_index_cleanup()
    with _index_cleanup_lock:
        if _pending_index_deletions:
            raise RuntimeError("Upload index cleanup is pending; retry the upload after storage recovers")
    if not files or len({file.file_id for file in files}) != len(files):
        raise ValueError("Upload index requires a nonempty set of distinct file IDs")
    if len({file.source_uri for file in files}) != len(files):
        raise ValueError("Upload sources must have distinct identities")
    if not session_id or not generation:
        raise ValueError("Upload index requires a session and generation")
    session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
    generation_hash = hashlib.sha256(generation.encode("utf-8")).hexdigest()[:24]
    collection_name = f"upload-{session_hash}-{generation_hash}"
    documents: list[ChunkedDocument] = []
    try:
        new_formats = [file for file in files if is_document_upload(file.name)]
        converted = {}
        if new_formats:
            if ingestion is None or ingestion.converter is None:
                raise IngestionError("DOCUMENT_CONVERTER_UNAVAILABLE", "문서 변환 기능을 활성화해 주세요.", retryable=True)
            ingestion.check_deadline()
            parsed = ingestion.converter.convert_many(new_formats, policy=ingestion.policy,
                workspace=ingestion.workspace,
                cache_dir=ingestion.cache_dir / "conversions" if ingestion.cache_dir is not None else None,
                deadline=ingestion.deadline)
            if len(parsed) != len(new_formats):
                raise IngestionError("DOCUMENT_ADAPTER_ERROR", "변환 결과의 파일 개수가 일치하지 않습니다.")
            converted = {file.file_id: document for file, document in zip(new_formats, parsed, strict=True)}
        document_chunks = 0
        for file in files:
            try:
                if file.file_id in converted:
                    parsed = converted[file.file_id]
                    if (parsed.snapshot.content_hash != file.content_hash or parsed.snapshot.source_uri != file.source_uri
                            or parsed.snapshot.capture_scope != "full_document"):
                        raise IngestionError("DOCUMENT_ADAPTER_ERROR", "변환 결과가 원본과 일치하지 않습니다.", file_name=file.name)
                    chunked = chunk_parsed_document(parsed, chunk_size=UPLOAD_CHUNK_SIZE, chunk_overlap=UPLOAD_CHUNK_OVERLAP)
                    documents.append(chunked)
                    if not chunked.chunks:
                        raise IngestionError("DOCUMENT_NO_SEARCHABLE_CONTENT", "검색 가능한 본문이나 표가 없습니다.", file_name=file.name)
                    document_chunks += len(chunked.chunks)
                    if document_chunks > ingestion.max_chunks or any(len(chunk.page_content.encode("utf-8")) > 8000 for chunk in chunked.chunks):
                        raise IngestionError("DOCUMENT_LIMIT_EXCEEDED", "문서 청크 수 또는 표 한 행의 크기 한도를 초과했습니다.", file_name=file.name)
                    for element in chunked.parsed.elements:
                        element.metadata["file_id"] = file.file_id
                    for chunk in chunked.chunks:
                        chunk.metadata["file_id"] = file.file_id
                else:
                    documents.append(_parse_upload(file))
            except IngestionError:
                raise
            except Exception as exc:
                raise ValueError(f"{file.name}: {exc}") from exc
    except BaseException:
        for document in documents:
            document.release()
        raise
    catalog = tuple(file.public_info() for file in files)
    return _index_documents(tuple(documents), collection_name=collection_name,
                            api_key=api_key, k=k, upload_files=catalog, ingestion=ingestion)
