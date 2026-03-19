from __future__ import annotations

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
    return RetrievalDiagnostic(
        tool=str(value.get("tool") or "").strip(),
        route=str(value.get("route") or "").strip(),
        status=str(value.get("status") or "").strip(),
        message=str(value.get("message") or ""),
        query=str(value.get("query") or ""),
        attempt=max(0, attempt),
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
