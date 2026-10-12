from __future__ import annotations

from dataclasses import dataclass

from src.core.contracts.debug import PlannerDiagnostic, PlannerStatus
from src.core.planner_schema import PlannerOutput


@dataclass(slots=True)
class PlannerDecision:
    output: PlannerOutput
    diagnostics: PlannerDiagnostic
    guided_followup: str | None = None
    status: PlannerStatus = "llm"
