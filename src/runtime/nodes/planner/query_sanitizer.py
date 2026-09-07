from __future__ import annotations

import re
from src.core.contracts.debug import RetryState
from src.core.planner_schema import PLANNER_ROUTES, PlannerOutput


def sanitize_retrieval_query(
    *,
    route: str,
    query: str,
    retry_context: RetryState | None = None,
) -> str:
    """Normalize whitespace while preserving the planner's source-specific meaning."""
    if route not in PLANNER_ROUTES:
        raise ValueError(f"Unsupported retrieval route: {route}")
    return " ".join(str(query or "").split())


def sanitize_planner_output_queries(
    planner_output: PlannerOutput,
    *,
    user_input: str,
    retry_context: RetryState | None = None,
    constraint_context: str | None = None,
) -> PlannerOutput:
    if not planner_output.use_retrieval or not planner_output.tasks:
        return planner_output
    sanitized_tasks = []
    source_request = constraint_context or user_input
    for task in planner_output.tasks:
        query = task.query or user_input
        grounded_aspects = []
        for aspect in task.requirement.aspects:
            # A planner can propose useful search terms, but it cannot invent
            # mandatory parameter values not requested in the user's dialogue.
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(aspect)}(?![A-Za-z0-9_])"
            if re.search(pattern, source_request, re.I):
                grounded_aspects.append(aspect)
            # The original query may keep model-proposed search vocabulary.
            # Only literal user constraints can become mandatory evidence anchors.
        sanitized_tasks.append(task.model_copy(update={
            "requirement": task.requirement.model_copy(update={"aspects": grounded_aspects}),
            "query": sanitize_retrieval_query(
                route=task.route,
                query=query or user_input,
                retry_context=retry_context,
            ),
        }))
    return planner_output.model_copy(update={"tasks": sanitized_tasks})
