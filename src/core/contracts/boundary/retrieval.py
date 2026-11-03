from __future__ import annotations

from typing import Any

from src.core.contracts.debug import RetrievalDiagnostic
from src.core.contracts.graph_state import GraphState, RetrievalState


def parse_retrieval_diagnostic(value: Any) -> RetrievalDiagnostic:
    if isinstance(value, RetrievalDiagnostic):
        return value
    return RetrievalDiagnostic.model_validate(value)


def parse_retrieval_diagnostics(value: Any) -> list[RetrievalDiagnostic]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("retrieval diagnostics must be a list")
    return [parse_retrieval_diagnostic(item) for item in value]


def normalize_retrieval_diagnostic_observation(
    value: Any,
) -> tuple[list[RetrievalDiagnostic], list[str]]:
    """Read observed diagnostics without treating invalid evidence as absence.

    Runtime state uses the typed parser above. HTTP diagnostics and policy
    evaluation need both safe values and the reasons any evidence
    was discarded, including a missing/null list. Execution identity and result
    matching remain the policy evaluator's responsibility.
    """
    label = "debug.retrieval_diagnostics"
    if not isinstance(value, list):
        return [], [f"{label} must be a list"]
    diagnostics: list[RetrievalDiagnostic] = []
    issues: list[str] = []
    for index, item in enumerate(value):
        if isinstance(item, RetrievalDiagnostic):
            item = item.model_dump(mode="python")
        if not isinstance(item, dict):
            issues.append(f"{label}[{index}] must be an object")
            continue
        item_issues = []
        for field in ("tool", "status"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                item_issues.append(f"{label}[{index}].{field} must be a non-empty string")
        invocation_id = item.get("invocation_id")
        if invocation_id is not None and (not isinstance(invocation_id, str) or not invocation_id.strip()):
            item_issues.append(f"{label}[{index}].invocation_id must be a non-empty string or null")
        if item_issues:
            issues.extend(item_issues)
            continue
        try:
            diagnostic = parse_retrieval_diagnostic(item)
        except (TypeError, ValueError, OverflowError) as exc:
            issues.append(f"{label}[{index}] invalid: {exc}")
        else:
            diagnostics.append(diagnostic)
    return diagnostics, issues


def get_retrieval_state(state: GraphState) -> RetrievalState:
    return state["retrieval"] if "retrieval" in state else RetrievalState()
