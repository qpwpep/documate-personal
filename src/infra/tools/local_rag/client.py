from __future__ import annotations

from typing import Any

from src.infra.chroma_store import build_openai_embeddings as build_openai_embeddings
from src.infra.tools._common import to_float_or_none


def search_with_raw_scores(vectorstore: Any, *, query: str, k: int) -> list[tuple[Any, float | None]]:
    if hasattr(vectorstore, "similarity_search_with_score"):
        return [
            (doc, to_float_or_none(score))
            for doc, score in vectorstore.similarity_search_with_score(query, k=k)
        ]
    if hasattr(vectorstore, "similarity_search"):
        return [(doc, None) for doc in vectorstore.similarity_search(query, k=k)]
    raise AttributeError("vectorstore does not support raw similarity search")
