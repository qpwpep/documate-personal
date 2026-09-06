from __future__ import annotations

from dataclasses import dataclass

from src.core.planner_schema import PlannerOutput


@dataclass(frozen=True, slots=True)
class SynthesisBudgetProfile:
    category: str
    max_tokens: int
    snippet_chars: int
    evidence_chars: int
    max_evidence_items: int


def resolve_synthesis_budget_profile(
    *, user_input: str, planner_output: PlannerOutput, synthesis_max_tokens: int,
) -> SynthesisBudgetProfile:
    routes = {task.route for task in planner_output.tasks} if planner_output.use_retrieval else set()
    hybrid = {"docs", "upload"}.issubset(routes)
    category = "hybrid" if hybrid else next(iter(routes), "general")
    # Saving or sending a researched answer does not reduce its evidence budget.
    return SynthesisBudgetProfile(
        category=category,
        max_tokens=max(1, int(synthesis_max_tokens)),
        snippet_chars=1800,
        evidence_chars=8000 if hybrid else 6000,
        max_evidence_items=8 if hybrid else 6,
    )


def compact_synthesis_budget_profile(profile: SynthesisBudgetProfile) -> SynthesisBudgetProfile:
    return SynthesisBudgetProfile(
        category=profile.category,
        max_tokens=max(1, profile.max_tokens // 2),
        snippet_chars=max(200, profile.snippet_chars // 2),
        evidence_chars=max(800, profile.evidence_chars // 2),
        max_evidence_items=profile.max_evidence_items,
    )
