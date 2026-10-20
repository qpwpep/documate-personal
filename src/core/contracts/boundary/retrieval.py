from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from src.core.contracts.debug import RetrievalDiagnostic
from src.core.contracts.graph_state import RetrievalState
from src.core.evidence import SearchHit, dedupe_search_hits


def parse_search_hit_message(content: str, *, errors: list[str]) -> list[SearchHit]:
    """Validate the current JSON message envelope at the serialization boundary."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        errors.append(f"search hits: invalid JSON ({exc})")
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("hits"), list):
        errors.append("search hits: expected an object containing a hits list")
        return []
    hits = []
    for index, item in enumerate(payload["hits"]):
        try:
            hits.append(SearchHit.model_validate(item))
        except (ValidationError, TypeError, ValueError) as exc:
            errors.append(f"search hits[{index}]: invalid item ({exc})")
    return dedupe_search_hits(hits)


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


def parse_retrieval_state(value: Any) -> RetrievalState:
    if isinstance(value, RetrievalState):
        return value
    if value is None:
        return RetrievalState()
    return RetrievalState.model_validate(value)


def get_retrieval_state(state: dict[str, Any]) -> RetrievalState:
    return parse_retrieval_state(state.get("retrieval"))
