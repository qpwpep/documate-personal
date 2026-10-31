from __future__ import annotations

from src.core.contracts import RetrievalDiagnostic
from src.core.llm_errors import ExecutionProblem, make_problem


def route_error_statuses(diagnostics: list[RetrievalDiagnostic]) -> set[str]:
    return {str(item.status or "").strip() for item in diagnostics if str(item.status or "").strip()}


def non_retryable_retrieval_problem(diagnostics: list[RetrievalDiagnostic]) -> ExecutionProblem | None:
    for diagnostic in diagnostics:
        if diagnostic.error_code in {"provider_configuration", "provider_rate_limited"}:
            return make_problem(diagnostic.error_code, "retrieval")
    return None
