from __future__ import annotations

from dataclasses import replace

from src.core.planner_schema import PlannerOutput
from src.runtime.nodes.retry import build_missing_upload_followup
from src.runtime.nodes.planner.models import PlannerDecision


def apply_retrieval_availability(
    decision: PlannerDecision,
    *,
    has_retriever: bool,
) -> PlannerDecision:
    """Check execution prerequisites without reinterpreting the requested sources."""
    required_routes = [task.route for task in decision.output.tasks]
    diagnostics = decision.diagnostics.model_copy(update={
        "intent_required": bool(required_routes),
        "required_routes": required_routes,
    })
    if "upload" in required_routes and not has_retriever:
        return replace(
            decision,
            output=PlannerOutput.fallback(),
            diagnostics=diagnostics.model_copy(update={
                "reason": "upload_retriever_missing",
                "override_applied": True,
                "override_reason": "upload_retriever_missing",
            }),
            guided_followup=build_missing_upload_followup(),
        )
    return replace(decision, diagnostics=diagnostics)
