from __future__ import annotations

import math
from typing import Any, Literal

from src.core.evidence import SearchHit


def build_retrieval_payload(
    *,
    tool: str,
    route: Literal["docs", "upload"],
    query: str,
    hits: list[SearchHit | dict[str, Any]] | None = None,
    status: Literal["success", "no_result", "error", "unavailable"] = "success",
    message: str = "",
    normalized_score: float | None = None,
    raw_score: float | None = None,
    provider_ms: int = 0,
    url_validation_ms: int = 0,
    post_filter_ms: int = 0,
    include_raw_content_requested: bool = False,
    result_count: int | None = None,
    provider_result_count: int | None = None,
    filtered_invalid_url_count: int = 0,
    filtered_path_prefix_count: int = 0,
    filtered_cross_domain_count: int = 0,
    filtered_http_error_count: int = 0,
    filtered_redirect_policy_count: int = 0,
    filtered_url_request_failed_count: int = 0,
    filtered_identifier_mismatch_count: int = 0,
    validated_url_count: int = 0,
    final_evidence_count: int | None = None,
    metric: str | None = None,
    score_direction: Literal["higher_is_better", "lower_is_better"] | None = None,
    warnings: list[str] | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    search_hits = [item if isinstance(item, SearchHit) else SearchHit.model_validate(item) for item in (hits or [])]
    resolved_metric = metric or ("provider_score" if route == "docs" else "l2")
    resolved_score_direction = (
        score_direction or ("higher_is_better" if route == "docs" else "lower_is_better")
    )
    resolved_normalized_score = normalized_score
    if resolved_normalized_score is None:
        scores = [
            item.score.normalized
            for item in search_hits
            if item.score.normalized is not None
        ]
        if scores:
            resolved_normalized_score = max(0.0, min(1.0, max(scores)))
    return {
        "hits": [item.model_dump(mode="json") for item in search_hits],
        "diagnostics": {
            "tool": tool,
            "route": route,
            "status": status,
            "message": message,
            "error_code": str(error_code or "").strip().upper() or None,
            "query": query,
            "evidence_count": len(search_hits),
            "metric": resolved_metric,
            "score_direction": resolved_score_direction,
            "normalized_score": resolved_normalized_score,
            "raw_score": raw_score,
            "provider_ms": max(0, int(provider_ms)),
            "url_validation_ms": max(0, int(url_validation_ms)),
            "post_filter_ms": max(0, int(post_filter_ms)),
            "include_raw_content_requested": bool(include_raw_content_requested),
            "result_count": len(search_hits) if result_count is None else max(0, int(result_count)),
            "provider_result_count": 0 if provider_result_count is None else max(0, int(provider_result_count)),
            "filtered_invalid_url_count": max(0, int(filtered_invalid_url_count)),
            "filtered_path_prefix_count": max(0, int(filtered_path_prefix_count)),
            "filtered_cross_domain_count": max(0, int(filtered_cross_domain_count)),
            "filtered_http_error_count": max(0, int(filtered_http_error_count)),
            "filtered_redirect_policy_count": max(0, int(filtered_redirect_policy_count)),
            "filtered_url_request_failed_count": max(0, int(filtered_url_request_failed_count)),
            "filtered_identifier_mismatch_count": max(0, int(filtered_identifier_mismatch_count)),
            "validated_url_count": max(0, int(validated_url_count)),
            "final_evidence_count": len(search_hits) if final_evidence_count is None else max(0, int(final_evidence_count)),
            "warnings": [str(item).strip() for item in (warnings or []) if str(item).strip()],
        },
    }


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
