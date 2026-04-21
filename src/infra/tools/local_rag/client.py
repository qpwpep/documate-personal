from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from src.infra.chroma_store import NOTEBOOK_COLLECTION_NAME, build_openai_embeddings, create_chroma_vectorstore
from src.infra.runtime_paths import get_local_rag_index_dir
from src.infra.tools._common import to_float_or_none


INDEX_PATH = get_local_rag_index_dir()


def load_chroma(openai_api_key: str, index_path: Path | None = None) -> Chroma:
    embeddings = build_openai_embeddings(openai_api_key)
    return create_chroma_vectorstore(
        embeddings=embeddings,
        persist_directory=index_path or INDEX_PATH,
        collection_name=NOTEBOOK_COLLECTION_NAME,
    )


def search_with_raw_scores(vectorstore: Any, *, query: str, k: int) -> list[tuple[Any, float | None]]:
    if hasattr(vectorstore, "similarity_search_with_score"):
        return [
            (doc, to_float_or_none(score))
            for doc, score in vectorstore.similarity_search_with_score(query, k=k)
        ]
    if hasattr(vectorstore, "similarity_search"):
        return [(doc, None) for doc in vectorstore.similarity_search(query, k=k)]
    raise AttributeError("vectorstore does not support raw similarity search")
