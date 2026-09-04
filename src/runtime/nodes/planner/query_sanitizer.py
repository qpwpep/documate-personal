from __future__ import annotations

from src.core.contracts.debug import RetryState
from src.core.planner_schema import PLANNER_ROUTES, PlannerOutput, RetrievalTask


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
) -> PlannerOutput:
    if not planner_output.use_retrieval or not planner_output.tasks:
        return planner_output
    sanitized_tasks = [
        RetrievalTask(
            route=task.route,
            query=sanitize_retrieval_query(
                route=task.route,
                query=task.query or user_input,
                retry_context=retry_context,
            ),
            k=task.k,
        )
        for task in planner_output.tasks
    ]
    return PlannerOutput(use_retrieval=True, tasks=sanitized_tasks)
