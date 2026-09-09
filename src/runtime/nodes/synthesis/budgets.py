from __future__ import annotations

from dataclasses import dataclass

from src.core.planner_schema import MAX_PLANNER_TASKS, PlannerOutput


@dataclass(frozen=True, slots=True)
class SynthesisBudgetProfile:
    category: str
    snippet_chars: int
    evidence_chars: int
    max_evidence_items: int


def resolve_synthesis_budget_profile(
    *, user_input: str, planner_output: PlannerOutput, snippet_char_limit: int,
) -> SynthesisBudgetProfile:
    routes = {task.route for task in planner_output.tasks} if planner_output.use_retrieval else set()
    hybrid = {"docs", "upload"}.issubset(routes)
    category = "hybrid" if hybrid else next(iter(routes), "general")
    # Saving or sending a researched answer does not reduce its evidence budget.
    return SynthesisBudgetProfile(
        category=category,
        snippet_chars=snippet_char_limit,
        evidence_chars=8000 if hybrid else 6000,
        max_evidence_items=MAX_PLANNER_TASKS,
    )


def compact_synthesis_budget_profile(
    profile: SynthesisBudgetProfile, *, snippet_char_limit: int,
) -> SynthesisBudgetProfile:
    return SynthesisBudgetProfile(
        category=profile.category,
        snippet_chars=min(profile.snippet_chars, snippet_char_limit),
        evidence_chars=max(800, profile.evidence_chars // 2),
        max_evidence_items=profile.max_evidence_items,
    )
