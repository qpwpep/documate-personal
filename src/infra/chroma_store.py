from __future__ import annotations

import math

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


CHROMA_DISTANCE_METRIC = "l2"
CHROMA_SCORE_DIRECTION = "lower_is_better"
CHROMA_COLLECTION_METADATA = {"hnsw:space": CHROMA_DISTANCE_METRIC}


def build_openai_embeddings(api_key: str | None) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )


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
