from __future__ import annotations

import math
from typing import Any

from src.core.sequence_utils import safe_list
from src.core.contracts.debug import RetrievalDiagnostic
from src.core.contracts.graph_state import RetrievalState


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
        query=str(value.get("query") or ""),
        attempt=max(0, attempt),
        evidence_count=max(0, evidence_count),
        metric=str(value.get("metric") or "").strip(),
        score_direction=score_direction,  # type: ignore[arg-type]
        normalized_score=normalized_score_value,
        raw_score=raw_score_value,
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
