from __future__ import annotations

from ...contracts.debug import PlannerDiagnostic, PlannerOverrideReason, PlannerStatus
from ...contracts.routes import ROUTE_ORDER
from ...planner_schema import PlannerOutput, RetrievalTask
from ..retry import build_missing_upload_followup
from .intents import detect_required_routes, is_upload_only_request
from .models import PlannerDecision, normalize_planner_diagnostics
from .query_sanitizer import sanitize_planner_output_queries


def apply_required_route_guardrail(
    *,
    planner_output: PlannerOutput,
    planner_status: PlannerStatus,
    planner_diagnostics: PlannerDiagnostic,
    user_input: str,
    has_retriever: bool,
) -> tuple[PlannerOutput, PlannerDiagnostic, str | None]:
    decision = build_required_route_guardrail_decision(
        planner_output=planner_output,
        planner_status=planner_status,
        planner_diagnostics=planner_diagnostics,
        user_input=user_input,
        has_retriever=has_retriever,
    )
    return decision.output, decision.diagnostics, decision.guided_followup


def build_required_route_guardrail_decision(
    *,
    planner_output: PlannerOutput,
    planner_status: PlannerStatus,
    planner_diagnostics: PlannerDiagnostic,
    user_input: str,
    has_retriever: bool,
) -> PlannerDecision:
    required_routes = detect_required_routes(user_input)
    diagnostics = normalize_planner_diagnostics(
        status=planner_status,
        reason=planner_diagnostics.reason,
        fallback_routes=planner_diagnostics.fallback_routes,
        intent_required=bool(required_routes),
        required_routes=required_routes,
        override_applied=planner_diagnostics.override_applied,
        override_reason=planner_diagnostics.override_reason,
    )

    if not required_routes:
        return PlannerDecision(output=planner_output, diagnostics=diagnostics, status=planner_status)

    if "upload" in required_routes and not has_retriever:
        return PlannerDecision(
            output=PlannerOutput.fallback(),
            diagnostics=diagnostics.model_copy(
                update={
                    "reason": "upload_retriever_missing",
                    "override_applied": True,
                    "override_reason": "upload_retriever_missing",
                }
            ),
            guided_followup=build_missing_upload_followup(),
            status=planner_status,
        )

    upload_only = is_upload_only_request(user_input)
    required_route_set = set(required_routes)
    existing_tasks = {task.route: task for task in planner_output.tasks}
    if upload_only:
        existing_tasks = {
            route: task for route, task in existing_tasks.items() if route in required_route_set
        }
    existing_routes = {task.route for task in planner_output.tasks} if planner_output.use_retrieval else set()
    if upload_only:
        existing_routes = {route for route in existing_routes if route in required_route_set}
    missing_required_routes = [route for route in required_routes if route not in existing_routes]

    override_reason: PlannerOverrideReason | None = None
    if required_routes and not planner_output.use_retrieval:
        override_reason = "missing_required_retrieval"
    elif missing_required_routes:
        override_reason = "missing_required_routes"

    if override_reason is None:
        return PlannerDecision(output=planner_output, diagnostics=diagnostics, status=planner_status)

    diagnostics = diagnostics.model_copy(
        update={
            "override_applied": True,
            "override_reason": override_reason,
            "reason": diagnostics.reason or override_reason,
        }
    )

    merged_tasks: list[RetrievalTask] = []
    for route in ROUTE_ORDER:
        if upload_only and route not in required_route_set:
            continue
        existing_task = existing_tasks.get(route)
        if existing_task is not None:
            merged_tasks.append(existing_task)
            continue
        if route in required_route_set:
            merged_tasks.append(RetrievalTask(route=route, query=str(user_input).strip(), k=4))

    sanitized_output = sanitize_planner_output_queries(
        PlannerOutput(use_retrieval=True, tasks=merged_tasks),
        user_input=user_input,
    )
    return PlannerDecision(
        output=sanitized_output,
        diagnostics=diagnostics,
        status=planner_status,
    )


def sanitize_planner_output(
    planner_output: PlannerOutput,
    *,
    has_retriever: bool,
    errors: list[str],
) -> PlannerOutput:
    tasks: list[RetrievalTask] = list(planner_output.tasks)
    if not has_retriever and any(task.route == "upload" for task in tasks):
        tasks = [task for task in tasks if task.route != "upload"]
        errors.append("planner: dropped upload route because retriever is unavailable")

    try:
        return PlannerOutput(
            use_retrieval=bool(planner_output.use_retrieval and tasks),
            tasks=tasks,
        )
    except Exception as exc:
        errors.append(f"planner: sanitized output validation failed ({exc})")
        return PlannerOutput.fallback()
