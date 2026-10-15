from __future__ import annotations

import math
from typing import Any, Literal

from src.core.contracts.debug import ErrorCode, RetrievalDiagnostic
from src.core.contracts.retrieval import RetrievalResult
from src.core.evidence import SearchHit


def build_retrieval_result(
    *,
    tool: str,
    route: Literal["docs", "upload"],
    query: str,
    hits: list[SearchHit] | None = None,
    status: Literal["success", "no_result", "error", "unavailable"] = "success",
    message: str = "",
    normalized_score: float | None = None,
    raw_score: float | None = None,
    provider_ms: int = 0,
    post_filter_ms: int = 0,
    include_raw_content_requested: bool = False,
    provider_result_count: int = 0,
    filtered_invalid_url_count: int = 0,
    filtered_path_prefix_count: int = 0,
    filtered_cross_domain_count: int = 0,
    filtered_identifier_mismatch_count: int = 0,
    metric: str | None = None,
    score_direction: Literal["higher_is_better", "lower_is_better"] | None = None,
    warnings: list[str] | None = None,
    error_code: ErrorCode | None = None,
) -> RetrievalResult:
    search_hits = hits if hits is not None else []
    if normalized_score is None:
        normalized_score = max((hit.score.normalized for hit in search_hits
                                if hit.score.normalized is not None), default=None)
    return RetrievalResult(
        hits=search_hits,
        diagnostics=RetrievalDiagnostic(
            tool=tool, route=route, status=status, message=message,
            error_code=error_code, query=query, evidence_count=len(search_hits),
            metric=metric or ("provider_score" if route == "docs" else "l2"),
            score_direction=score_direction or ("higher_is_better" if route == "docs" else "lower_is_better"),
            normalized_score=normalized_score, raw_score=raw_score,
            provider_ms=provider_ms, post_filter_ms=post_filter_ms,
            include_raw_content_requested=include_raw_content_requested,
            result_count=len(search_hits),
            provider_result_count=provider_result_count,
            filtered_invalid_url_count=filtered_invalid_url_count,
            filtered_path_prefix_count=filtered_path_prefix_count,
            filtered_cross_domain_count=filtered_cross_domain_count,
            filtered_identifier_mismatch_count=filtered_identifier_mismatch_count,
            final_evidence_count=len(search_hits),
            warnings=sorted(set(warnings or [])),
        ),
    )


def to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_relevance_score(
    value: Any,
    *,
    warnings: list[str] | None = None,
) -> tuple[float | None, float | None]:
    raw_score = to_float_or_none(value)
    if raw_score is None or not math.isfinite(raw_score):
        if value is not None and warnings is not None:
            warnings.append("invalid_relevance_score")
        return None, None

    normalized_score = raw_score
    if raw_score < 0.0 or raw_score > 1.0:
        normalized_score = max(0.0, min(1.0, raw_score))
        if warnings is not None:
            warnings.append("relevance_score_clamped")
    return normalized_score, raw_score
