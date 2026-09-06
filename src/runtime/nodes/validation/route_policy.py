from __future__ import annotations

from src.core.contracts import RetrievalDiagnostic


def route_error_statuses(diagnostics: list[RetrievalDiagnostic]) -> set[str]:
    return {str(item.status or "").strip() for item in diagnostics if str(item.status or "").strip()}
