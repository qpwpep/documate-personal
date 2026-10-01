from __future__ import annotations

from dataclasses import dataclass

from src.core.planner_schema import PlannerOutput, RetrievalTask


# Empirical selection policy, not measured model-context limits. The pairs are
# normal/compact sums of Python string lengths for newly retrieved excerpts.
_DEFAULT_TOTAL_CHARS = (6000, 3000)
_MIXED_ROUTE_TOTAL_CHARS = (8000, 4000)
_BASE_ITEM_CAPACITY = 8


@dataclass(frozen=True, slots=True)
class ExcerptLimits:
    """Configured per-excerpt character limits; settings owns defaults and validation."""

    normal_chars: int
    compact_chars: int


@dataclass(frozen=True, slots=True)
class RetrievedEvidenceBudget:
    """Selection ceilings for retrieved excerpts, excluding inherited citations.

    Character limits count Python string lengths, not bytes or model tokens.
    Prompt metadata, history, bound source answers and output tokens are outside
    this budget. Item capacity permits coverage attempts; it does not guarantee
    that every required passage fits the shared character limit.
    """

    max_excerpt_chars: int
    max_total_excerpt_chars: int
    max_items: int


def requirement_passage_targets(task: RetrievalTask) -> list[tuple[str | None, str | None]]:
    """Expand explicit files and literal aspects in their existing selection order."""
    return [(file_id, aspect)
            for file_id in task.requirement.file_ids or [None]
            for aspect in task.requirement.aspects or [None]]


def resolve_evidence_budgets(
    *, plan: PlannerOutput, limits: ExcerptLimits,
) -> tuple[RetrievedEvidenceBudget, RetrievedEvidenceBudget]:
    """Resolve normal and timeout-recovery budgets once from the retrieval plan.

    Saving or sending a researched answer does not reduce its evidence budget.
    Compact totals are fixed recovery policy, independently of normal totals.
    """
    routes = {task.route for task in plan.tasks} if plan.use_retrieval else set()
    normal_total, compact_total = (
        _MIXED_ROUTE_TOTAL_CHARS if {"docs", "upload"}.issubset(routes) else _DEFAULT_TOTAL_CHARS
    )
    required_passages = sum(len(requirement_passage_targets(task)) for task in plan.tasks)
    max_items = max(_BASE_ITEM_CAPACITY, required_passages)
    return (
        RetrievedEvidenceBudget(limits.normal_chars, normal_total, max_items),
        RetrievedEvidenceBudget(min(limits.normal_chars, limits.compact_chars), compact_total, max_items),
    )
