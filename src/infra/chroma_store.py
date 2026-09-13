from __future__ import annotations

import math
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


CHROMA_DISTANCE_METRIC = "l2"
CHROMA_SCORE_DIRECTION = "lower_is_better"
CHROMA_COLLECTION_METADATA = {"hnsw:space": CHROMA_DISTANCE_METRIC}


def build_openai_embeddings(api_key: str | None, *, timeout_seconds: float | None = None) -> OpenAIEmbeddings:
    options = {} if timeout_seconds is None else {"request_timeout": timeout_seconds, "max_retries": 0}
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        **options,
    )


def add_preembedded_documents(vectorstore: Chroma, documents: list, vectors: list[list[float]]) -> None:
    """Use Chroma's explicit-vector boundary without changing per-file embedding namespaces."""
    if len(documents) != len(vectors):
        raise ValueError("Embedding result count differs from document count")
    vectorstore._collection.upsert(ids=[uuid4().hex for _ in documents], embeddings=vectors,
        documents=[document.page_content for document in documents],
        metadatas=[document.metadata for document in documents])


def create_chroma_vectorstore(
    *,
    embeddings: OpenAIEmbeddings,
    collection_name: str,
) -> Chroma:
    return Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        collection_metadata=dict(CHROMA_COLLECTION_METADATA),
    )


def normalize_l2_distance(distance: float | None) -> float | None:
    if distance is None:
        return None
    if not math.isfinite(float(distance)):
        return None
    return max(0.0, min(1.0, 1.0 - (float(distance) / math.sqrt(2.0))))
