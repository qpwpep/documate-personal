from __future__ import annotations

from typing import Any

from src.core.answer_schema import AgentResponsePayloadModel
from src.core.contracts import GraphState, RetrievalDiagnostic
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state, parse_planner_output
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.contracts.routes import route_for_tool
from src.core.evidence import EvidenceItem, parse_evidence_payload
from src.core.planner_schema import PlannerOutput
from src.core.sequence_utils import slice_from_index
from src.runtime.nodes.validation.models import ValidationSnapshot


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


def collect_validation_snapshot(state: GraphState) -> tuple[ValidationSnapshot, list[str]]:
    local_errors: list[str] = []
    parse_errors: list[str] = []
    runtime = get_runtime_state(state)
    planner = get_planner_state(state)
    retrieval = get_retrieval_state(state)
    response = get_response_state(state)
    debug = get_debug_state(state)
    retry_context = get_retry_state(state)

    planner_output = parse_planner_output(planner.output, local_errors)
    evidence_start_index = int(retry_context.evidence_start_index)
    retrieval_error_start_index = int(retry_context.retrieval_error_start_index)
    retrieval_diagnostic_start_index = int(retry_context.retrieval_diagnostic_start_index)

    current_attempt_evidence_payload = slice_from_index(
        retrieval.evidence_log,
        evidence_start_index,
    )
    parsed_evidence = coerce_evidence_list(
        parse_evidence_payload(
            current_attempt_evidence_payload,
            context="retrieved_evidence",
            errors=parse_errors,
        )
    )
    local_errors.extend(parse_errors)

    current_attempt_retrieval_errors = [
        str(error)
        for error in slice_from_index(
            debug.retrieval_errors,
            retrieval_error_start_index,
        )
        if str(error).strip()
    ]
    current_attempt_retrieval_diagnostics = [
        item
        for item in slice_from_index(
            debug.retrieval_diagnostics,
            retrieval_diagnostic_start_index,
        )
        if item is not None
    ]

    snapshot = build_validation_snapshot(
        user_input=runtime.user_input,
        planner_output=planner_output,
        parsed_evidence=parsed_evidence,
        current_attempt_retrieval_errors=[*current_attempt_retrieval_errors, *parse_errors],
        current_attempt_retrieval_diagnostics=current_attempt_retrieval_diagnostics,
        response_payload=response.payload,
    )
    return snapshot, local_errors
