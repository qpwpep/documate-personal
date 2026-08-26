from __future__ import annotations

import math
from typing import Any

from src.core.sequence_utils import safe_list
from src.core.contracts.debug import ErrorCode, RetrievalDiagnostic
from src.core.contracts.graph_state import RetrievalState


_ERROR_CODES = set(ErrorCode.__args__)  # type: ignore[attr-defined]


def _parse_error_code(value: Any) -> str | None:
    code = str(value or "").strip().upper()
    return code if code in _ERROR_CODES else None


def _parse_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, default)


def parse_retrieval_diagnostic(value: Any) -> RetrievalDiagnostic | None:
    if isinstance(value, RetrievalDiagnostic):
        return value
    if not isinstance(value, dict):
        return None
    attempt = _parse_non_negative_int(value.get("attempt", 0), default=0)
    result_count = _parse_non_negative_int(value.get("result_count", 0), default=0)
    evidence_count = _parse_non_negative_int(
        value.get("evidence_count", result_count),
        default=result_count,
    )
    provider_result_count = _parse_non_negative_int(value.get("provider_result_count", 0), default=0)
    filtered_invalid_url_count = _parse_non_negative_int(value.get("filtered_invalid_url_count", 0), default=0)
    filtered_path_prefix_count = _parse_non_negative_int(value.get("filtered_path_prefix_count", 0), default=0)
    filtered_cross_domain_count = _parse_non_negative_int(value.get("filtered_cross_domain_count", 0), default=0)
    filtered_http_error_count = _parse_non_negative_int(value.get("filtered_http_error_count", 0), default=0)
    filtered_redirect_policy_count = _parse_non_negative_int(value.get("filtered_redirect_policy_count", 0), default=0)
    filtered_url_request_failed_count = _parse_non_negative_int(value.get("filtered_url_request_failed_count", 0), default=0)
    filtered_identifier_mismatch_count = _parse_non_negative_int(value.get("filtered_identifier_mismatch_count", 0), default=0)
    validated_url_count = _parse_non_negative_int(value.get("validated_url_count", 0), default=0)
    final_evidence_count = _parse_non_negative_int(value.get("final_evidence_count", evidence_count), default=evidence_count)
    provider_ms = _parse_non_negative_int(value.get("provider_ms", 0), default=0)
    url_validation_ms = _parse_non_negative_int(value.get("url_validation_ms", 0), default=0)
    post_filter_ms = _parse_non_negative_int(value.get("post_filter_ms", 0), default=0)
    normalized_score = value.get("normalized_score")
    raw_score = value.get("raw_score")
    try:
        normalized_score_value = (
            None if normalized_score is None else max(0.0, min(1.0, float(normalized_score)))
        )
    except (TypeError, ValueError):
        normalized_score_value = None
    try:
        raw_score_value = None if raw_score is None else float(raw_score)
    except (TypeError, ValueError):
        raw_score_value = None
    if raw_score_value is not None and not math.isfinite(raw_score_value):
        raw_score_value = None
    warnings = value.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    score_direction = str(value.get("score_direction") or "").strip()
    if score_direction not in {"higher_is_better", "lower_is_better"}:
        score_direction = ""
    return RetrievalDiagnostic(
        tool=str(value.get("tool") or "").strip(),
        route=str(value.get("route") or "").strip(),
        status=str(value.get("status") or "").strip(),
        message=str(value.get("message") or ""),
        error_code=_parse_error_code(value.get("error_code")),  # type: ignore[arg-type]
        query=str(value.get("query") or ""),
        attempt=attempt,
        evidence_count=evidence_count,
        metric=str(value.get("metric") or "").strip(),
        score_direction=score_direction,  # type: ignore[arg-type]
        normalized_score=normalized_score_value,
        raw_score=raw_score_value,
        provider_ms=provider_ms,
        url_validation_ms=url_validation_ms,
        post_filter_ms=post_filter_ms,
        include_raw_content_requested=bool(value.get("include_raw_content_requested", False)),
        result_count=result_count,
        provider_result_count=provider_result_count,
        filtered_invalid_url_count=filtered_invalid_url_count,
        filtered_path_prefix_count=filtered_path_prefix_count,
        filtered_cross_domain_count=filtered_cross_domain_count,
        filtered_http_error_count=filtered_http_error_count,
        filtered_redirect_policy_count=filtered_redirect_policy_count,
        filtered_url_request_failed_count=filtered_url_request_failed_count,
        filtered_identifier_mismatch_count=filtered_identifier_mismatch_count,
        validated_url_count=validated_url_count,
        final_evidence_count=final_evidence_count,
        warnings=[str(item).strip() for item in warnings if str(item).strip()],
    )


def parse_retrieval_diagnostics(value: Any) -> list[RetrievalDiagnostic]:
    if not isinstance(value, list):
        return []
    diagnostics: list[RetrievalDiagnostic] = []
    for item in value:
        diagnostic = parse_retrieval_diagnostic(item)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return diagnostics


def parse_retrieval_state(value: Any) -> RetrievalState:
    if isinstance(value, RetrievalState):
        return value
    if not isinstance(value, dict):
        return RetrievalState()
    evidence_log = value.get("evidence_log")
    return RetrievalState(
        evidence_log=[
            item
            for item in safe_list(evidence_log)
            if isinstance(item, dict)
        ]
    )


def get_retrieval_state(state: dict[str, Any]) -> RetrievalState:
    return parse_retrieval_state(state.get("retrieval"))
