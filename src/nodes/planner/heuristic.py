from __future__ import annotations

from ...contracts.debug import PlannerDiagnostic
from ...contracts.routes import ROUTE_ORDER
from ...planner_schema import PlannerOutput, RetrievalTask
from ...prompts import has_explicit_docs_intent, has_explicit_local_intent, needs_search
from ..retry import build_missing_upload_followup
from .intents import has_upload_route_intent, needs_upload_followup
from .models import PlannerDecision, normalize_planner_diagnostics
from .query_sanitizer import sanitize_planner_output_queries


def build_heuristic_planner_output(
    *,
    user_input: str,
    has_retriever: bool,
) -> tuple[PlannerOutput, PlannerDiagnostic, str | None]:
    decision = build_heuristic_planner_decision(
        user_input=user_input,
        has_retriever=has_retriever,
    )
    return decision.output, decision.diagnostics, decision.guided_followup


def build_heuristic_planner_decision(
    *,
    user_input: str,
    has_retriever: bool,
) -> PlannerDecision:
    trimmed_query = str(user_input or "").strip()
    routes: list[str] = []
    upload_route_intent = has_upload_route_intent(user_input)
    explicit_docs_intent = has_explicit_docs_intent(user_input)
    explicit_local_intent = has_explicit_local_intent(user_input)
    guided_followup: str | None = None

    if (upload_route_intent and explicit_docs_intent) or (
        not upload_route_intent and needs_search(user_input)
    ):
        routes.append("docs")

    if has_retriever and upload_route_intent:
        routes.append("upload")
    elif upload_route_intent and needs_upload_followup(user_input):
        guided_followup = build_missing_upload_followup()

    if explicit_local_intent and not explicit_docs_intent and not upload_route_intent:
        routes.append("local")

    unique_routes = [route for route in ROUTE_ORDER if route in routes]
    if unique_routes:
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route=route, query=trimmed_query, k=4) for route in unique_routes],
        )
        planner_output = sanitize_planner_output_queries(planner_output, user_input=trimmed_query)
        return PlannerDecision(
            output=planner_output,
            diagnostics=normalize_planner_diagnostics(
                status="heuristic_fallback",
                reason="planner_failed_or_invalid",
                fallback_routes=unique_routes,
            ),
            guided_followup=guided_followup,
            status="heuristic_fallback",
        )

    return PlannerDecision(
        output=PlannerOutput.fallback(),
        diagnostics=normalize_planner_diagnostics(
            status="fallback_no_routes",
            reason="planner_failed_or_invalid",
            fallback_routes=[],
        ),
        guided_followup=guided_followup,
        status="fallback_no_routes",
    )
