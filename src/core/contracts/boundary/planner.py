from __future__ import annotations

from typing import Any

from src.core.contracts.debug import PlannerDiagnostic
from src.core.contracts.graph_state import PlannerState


def parse_planner_diagnostic(value: Any) -> PlannerDiagnostic | None:
    if value is None:
        return None
    return PlannerDiagnostic.model_validate(value)


def parse_planner_state(value: Any) -> PlannerState:
    if value is None:
        return PlannerState()
    return PlannerState.model_validate(value)


def get_planner_state(state: dict[str, Any]) -> PlannerState:
    return parse_planner_state(state.get("planner"))
