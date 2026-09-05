from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from src.core.latency import elapsed_ms
from src.infra.chroma_store import CHROMA_DISTANCE_METRIC, CHROMA_SCORE_DIRECTION
from src.infra.tools._common import build_retrieval_payload
from src.infra.tools.local_rag import client
from src.infra.tools.local_rag.ranking import rank_retrieval_rows
from src.infra.tools.local_rag.serialization import build_local_evidence_bundle


def _build_search_payload(
    *,
    query: str,
    docs_with_scores: list[tuple[Any, float | None]],
    provider_ms: int = 0,
) -> dict[str, Any]:
    post_started = time.perf_counter()
    ranked_rows = rank_retrieval_rows(docs_with_scores, query=query)
    evidence, normalized_scores, raw_scores, retrieval_warnings = build_local_evidence_bundle(
        ranked_rows,
        query=query,
        tool_name="upload_search",
        default_source="uploaded",
    )
    post_filter_ms = elapsed_ms(post_started, time.perf_counter())
    return build_retrieval_payload(
        tool="upload_search",
        route="upload",
        query=query,
        evidence=evidence,
        status="success" if evidence else "no_result",
        message="" if evidence else "no uploaded file evidence found",
        normalized_score=max(normalized_scores) if normalized_scores else None,
        raw_score=min(raw_scores) if raw_scores else None,
        provider_ms=provider_ms,
        post_filter_ms=post_filter_ms,
        metric=CHROMA_DISTANCE_METRIC,
        score_direction=CHROMA_SCORE_DIRECTION,
        warnings=retrieval_warnings,
    )


def build_upload_search_tool() -> Callable[..., dict[str, Any]]:
    def upload_search(
        query: str,
        k: int = 4,
        retriever: Any = None,
    ) -> dict[str, Any]:
        if retriever is None:
            return build_retrieval_payload(
                tool="upload_search",
                route="upload",
                query=query,
                status="unavailable",
                message="upload retriever is unavailable; upload a .py or .ipynb file first",
            )

        docs_with_scores: list[tuple[Any, float | None]] = []
        provider_ms = 0
        provider_started = time.perf_counter()
        try:
            vectorstore = getattr(retriever, "vectorstore", None)
            provider_started = time.perf_counter()
            if vectorstore is not None:
                docs_with_scores = client.search_with_raw_scores(vectorstore, query=query, k=k)
            else:
                docs = retriever.invoke(query)
                docs_with_scores = [(doc, None) for doc in docs]
            provider_ms += elapsed_ms(provider_started, time.perf_counter())
        except Exception as exc:
            provider_ms += elapsed_ms(provider_started, time.perf_counter())
            return build_retrieval_payload(
                tool="upload_search",
                route="upload",
                query=query,
                status="error",
                message=f"uploaded file retrieval failed ({exc})",
                provider_ms=provider_ms,
                error_code="LOCAL_RAG_FAILED",
            )

        return _build_search_payload(
            query=query,
            docs_with_scores=docs_with_scores,
            provider_ms=provider_ms,
        )

    return upload_search
