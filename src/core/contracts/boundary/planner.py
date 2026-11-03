from __future__ import annotations

from typing import Any

from src.core.contracts.debug import PlannerDiagnostic
from src.core.contracts.graph_state import GraphState, PlannerState


def parse_planner_diagnostic(value: Any) -> PlannerDiagnostic | None:
    if value is None:
        return None
    return PlannerDiagnostic.model_validate(value)


def get_planner_state(state: GraphState) -> PlannerState:
    return state["planner"] if "planner" in state else PlannerState()
