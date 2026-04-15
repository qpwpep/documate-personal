from __future__ import annotations

from typing import Any

from ...answer_schema import AgentResponsePayloadModel
from ...contracts import RetrievalDiagnostic
from ...contracts.routes import route_for_tool
from ...evidence import EvidenceItem
from ...planner_schema import PlannerOutput
from .models import ValidationSnapshot


def coerce_evidence_list(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [item for item in items if isinstance(item, EvidenceItem)]


def route_for_item_tool(tool_name: str) -> str:
    return route_for_tool(str(tool_name or ""))


def detect_missing_route_coverage(
    *,
    required_routes: list[str],
    valid_claims: list[Any],
    route_by_source_id: dict[str, str],
) -> list[str]:
    covered_routes: set[str] = set()
    for claim in valid_claims:
        for source_id in getattr(claim, "evidence_ids", []) or []:
            route = route_by_source_id.get(str(source_id or "").strip(), "")
            if route:
                covered_routes.add(route)
    return [route for route in required_routes if route not in covered_routes]


def build_validation_snapshot(
    *,
    user_input: str,
    planner_output: PlannerOutput,
    parsed_evidence: list[EvidenceItem],
    current_attempt_retrieval_errors: list[str],
    current_attempt_retrieval_diagnostics: list[RetrievalDiagnostic],
    response_payload: AgentResponsePayloadModel | None,
) -> ValidationSnapshot:
    retrieval_required = bool(planner_output.use_retrieval and planner_output.tasks)
    evidence_by_route: dict[str, list[EvidenceItem]] = {"docs": [], "upload": [], "local": []}
    for item in parsed_evidence:
        route = route_for_item_tool(item.tool)
        if route:
            evidence_by_route.setdefault(route, []).append(item)

    diagnostics_by_route: dict[str, list[RetrievalDiagnostic]] = {"docs": [], "upload": [], "local": []}
    for item in current_attempt_retrieval_diagnostics:
        route = str(item.route or "").strip()
        if route:
            diagnostics_by_route.setdefault(route, []).append(item)

    required_routes = [task.route for task in planner_output.tasks] if retrieval_required else []
    return ValidationSnapshot(
        user_input=user_input,
        planner_output=planner_output,
        retrieval_required=retrieval_required,
        parsed_evidence=parsed_evidence,
        current_attempt_retrieval_errors=current_attempt_retrieval_errors,
        current_attempt_retrieval_diagnostics=current_attempt_retrieval_diagnostics,
        response_payload=response_payload,
        evidence_by_route=evidence_by_route,
        diagnostics_by_route=diagnostics_by_route,
        required_routes=required_routes,
    )
