from __future__ import annotations

import math
from typing import Any

from ...sequence_utils import safe_list
from ..debug import RetrievalDiagnostic
from ..graph_state import RetrievalState


def parse_retrieval_diagnostic(value: Any) -> RetrievalDiagnostic | None:
    if isinstance(value, RetrievalDiagnostic):
        return value
    if not isinstance(value, dict):
        return None
    try:
        attempt = int(value.get("attempt", 0) or 0)
    except (TypeError, ValueError):
        attempt = 0
    try:
        result_count = int(value.get("result_count", 0) or 0)
    except (TypeError, ValueError):
        result_count = 0
    try:
        evidence_count = int(value.get("evidence_count", result_count) or result_count)
    except (TypeError, ValueError):
        evidence_count = result_count
    relevance_score = value.get("relevance_score")
    raw_relevance_score = value.get("raw_relevance_score")
    avg_score = value.get("avg_score")
    max_score = value.get("max_score")
    normalized_score = value.get("normalized_score")
    try:
        normalized_relevance = (
            None if relevance_score is None else max(0.0, min(1.0, float(relevance_score)))
        )
    except (TypeError, ValueError):
        normalized_relevance = None
    try:
        raw_relevance = None if raw_relevance_score is None else float(raw_relevance_score)
    except (TypeError, ValueError):
        raw_relevance = None
    if raw_relevance is not None and not math.isfinite(raw_relevance):
        raw_relevance = None
    try:
        normalized_avg_score = None if avg_score is None else max(0.0, min(1.0, float(avg_score)))
    except (TypeError, ValueError):
        normalized_avg_score = None
    try:
        normalized_max_score = None if max_score is None else max(0.0, min(1.0, float(max_score)))
    except (TypeError, ValueError):
        normalized_max_score = None
    try:
        normalized_score_value = (
            None if normalized_score is None else max(0.0, min(1.0, float(normalized_score)))
        )
    except (TypeError, ValueError):
        normalized_score_value = None
    warnings = value.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    return RetrievalDiagnostic(
        tool=str(value.get("tool") or "").strip(),
        route=str(value.get("route") or "").strip(),
        status=str(value.get("status") or "").strip(),
        message=str(value.get("message") or ""),
        query=str(value.get("query") or ""),
        attempt=max(0, attempt),
        evidence_count=max(0, evidence_count),
        avg_score=normalized_avg_score,
        max_score=normalized_max_score,
        normalized_score=normalized_score_value,
        relevance_score=normalized_relevance,
        raw_relevance_score=raw_relevance,
        score=normalized_score_value if normalized_score_value is not None else normalized_relevance,
        result_count=max(0, result_count),
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
