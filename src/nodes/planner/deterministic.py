from __future__ import annotations

from ...planner_schema import PlannerOutput, RetrievalTask
from ...prompts import (
    has_explicit_docs_intent,
    has_explicit_local_intent,
    needs_save,
    needs_slack,
)
from ..actions import is_action_only_request
from ..retry import build_missing_upload_followup
from .intents import has_upload_route_intent
from .models import PlannerDecision, normalize_planner_diagnostics
from .query_sanitizer import sanitize_planner_output_queries


def build_deterministic_planner_decision(
    *,
    user_input: str,
    has_retriever: bool,
) -> PlannerDecision | None:
    docs_intent = has_explicit_docs_intent(user_input)
    upload_intent = has_upload_route_intent(user_input)
    local_intent = has_explicit_local_intent(user_input) and not docs_intent and not upload_intent
    action_only = is_action_only_request(user_input) or (
        (needs_save(user_input) or needs_slack(user_input))
        and not docs_intent
        and not upload_intent
        and not local_intent
    )

    if action_only:
        return PlannerDecision(
            output=PlannerOutput.fallback(),
            diagnostics=normalize_planner_diagnostics(
                status="deterministic",
                reason="action_only",
                fallback_routes=[],
            ),
            status="deterministic",
        )

    routes: list[str] = []
    if docs_intent and upload_intent:
        routes = ["docs", "upload"]
    elif upload_intent:
        routes = ["upload"]
    elif docs_intent:
        routes = ["docs"]
    elif local_intent:
        routes = ["local"]
    else:
        return None

    if "upload" in routes and not has_retriever:
        return PlannerDecision(
            output=PlannerOutput.fallback(),
            diagnostics=normalize_planner_diagnostics(
                status="deterministic",
                reason="upload_retriever_missing",
                fallback_routes=[],
                intent_required=True,
                required_routes=routes,
                override_applied=True,
                override_reason="upload_retriever_missing",
            ),
            guided_followup=build_missing_upload_followup(),
            status="deterministic",
        )

    planner_output = PlannerOutput(
        use_retrieval=True,
        tasks=[RetrievalTask(route=route, query=user_input.strip(), k=4) for route in routes],
    )
    planner_output = sanitize_planner_output_queries(planner_output, user_input=user_input)
    return PlannerDecision(
        output=planner_output,
        diagnostics=normalize_planner_diagnostics(
            status="deterministic",
            reason=None,
            fallback_routes=routes,
            intent_required=True,
            required_routes=routes,
        ),
        status="deterministic",
    )
