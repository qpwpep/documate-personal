from __future__ import annotations

from dataclasses import dataclass

from src.core.contracts.debug import PlannerDiagnostic, PlannerOverrideReason, PlannerStatus
from src.core.contracts.routes import ROUTE_ORDER
from src.core.planner_schema import PlannerOutput


@dataclass(slots=True)
class PlannerDecision:
    output: PlannerOutput
    diagnostics: PlannerDiagnostic
    guided_followup: str | None = None
    status: PlannerStatus = "llm"


def normalize_planner_diagnostics(
    *,
    status: PlannerStatus,
    reason: str | None = None,
    fallback_routes: list[str] | None = None,
    intent_required: bool = False,
    required_routes: list[str] | None = None,
    override_applied: bool = False,
    override_reason: PlannerOverrideReason | None = None,
) -> PlannerDiagnostic:
    return PlannerDiagnostic(
        status=status,
        reason=reason,
        fallback_routes=list(fallback_routes or []),
        intent_required=bool(intent_required),
        required_routes=[route for route in ROUTE_ORDER if route in set(required_routes or [])],
        override_applied=bool(override_applied),
        override_reason=override_reason,
    )
