from __future__ import annotations

from typing import Any

from src.core.contracts.debug import PlannerDiagnostic, normalize_recorded_routes
from src.core.contracts.graph_state import PlannerState


def parse_planner_diagnostic(value: Any) -> PlannerDiagnostic | None:
    if value is None:
        return None
    if isinstance(value, PlannerDiagnostic):
        return value
    if not isinstance(value, dict):
        return None

    fallback_routes = value.get("fallback_routes")
    required_routes = value.get("required_routes")
    status = value.get("status")
    reason = value.get("reason")
    override_reason = value.get("override_reason")
    planner_warnings = value.get("planner_warnings")
    if override_reason not in {
        "missing_required_retrieval",
        "missing_required_routes",
        "upload_retriever_missing",
    }:
        override_reason = None

    return PlannerDiagnostic(
        status=str(status) if status is not None else "",
        reason=str(reason) if reason is not None else None,
        fallback_routes=normalize_recorded_routes(fallback_routes) if isinstance(fallback_routes, list) else [],
        intent_required=bool(value.get("intent_required", False)),
        required_routes=normalize_recorded_routes(required_routes) if isinstance(required_routes, list) else [],
        override_applied=bool(value.get("override_applied", False)),
        override_reason=override_reason,
        planner_warnings=[
            str(warning).strip()
            for warning in planner_warnings
            if str(warning).strip()
        ]
        if isinstance(planner_warnings, list)
        else [],
    )


def parse_planner_state(value: Any) -> PlannerState:
    if value is None:
        return PlannerState()
    return PlannerState.model_validate(value)


def get_planner_state(state: dict[str, Any]) -> PlannerState:
    return parse_planner_state(state.get("planner"))
