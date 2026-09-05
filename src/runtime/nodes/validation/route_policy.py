from __future__ import annotations

from src.core.contracts import RetrievalDiagnostic
from src.core.evidence import EvidenceItem


def route_score_avg(items: list[EvidenceItem]) -> float | None:
    scores = [float(item.score) for item in items if item.score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def route_error_statuses(diagnostics: list[RetrievalDiagnostic]) -> set[str]:
    return {
        str(item.status or "").strip()
        for item in diagnostics
        if str(item.status or "").strip()
    }
