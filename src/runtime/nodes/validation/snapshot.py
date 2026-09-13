from __future__ import annotations

from src.core.answer_schema import AnswerResponse, iter_content_units
from src.core.contracts import GraphState, RetrievalDiagnostic
from src.core.contracts.provenance import AnswerSource, BodyKind
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state, parse_planner_output
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.evidence import EvidenceRef, SearchHit
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import RequestContract
from src.core.sequence_utils import slice_from_index
from src.runtime.nodes.validation.models import ValidationSnapshot
from src.runtime.nodes.synthesis.evidence_selection import missing_evidence_requirement_ids, tasks_for_hit


def detect_missing_route_coverage(
    *, required_routes: list[str], result: AnswerResponse,
    evidence_packet: list[EvidenceRef], valid_unit_paths: set[str],
) -> list[str]:
    route_by_id = {item.id: item.route for item in evidence_packet}
    covered_routes = {
        route_by_id[ref]
        for path, unit in iter_content_units(result.content)
        if path in valid_unit_paths
        for ref in unit.refs
        if ref in route_by_id
    }
    return [route for route in required_routes if route not in covered_routes]


def detect_missing_requirement_coverage(
    *, snapshot: ValidationSnapshot, result: AnswerResponse, valid_unit_paths: set[str],
) -> list[str]:
    cited = {ref for path, unit in iter_content_units(result.content) if path in valid_unit_paths for ref in unit.refs}
    return missing_evidence_requirement_ids(
        snapshot.planner_output,
        [item for item in snapshot.evidence_packet if item.id in cited],
        snapshot.evidence_requirement_map,
        strict=snapshot.normal_evidence_missing_requirement_ids is not None,
    )


def detect_packet_coverage_gaps(snapshot: ValidationSnapshot) -> tuple[list[str], list[str]]:
    """Separate missing retrieved anchors from requirements lost while selecting the packet."""
    # A known normal-packet result distinguishes current preparation from legacy
    # untagged packets, even when selection removed every source association.
    strict = snapshot.normal_evidence_missing_requirement_ids is not None
    packet_missing = missing_evidence_requirement_ids(snapshot.planner_output, snapshot.evidence_packet,
                                                     snapshot.evidence_requirement_map, strict=strict)
    retrieved_map: dict[str, list[str]] = {}
    for hit in snapshot.parsed_hits:
        associated = retrieved_map.setdefault(hit.evidence.id, [])
        associated.extend(task.requirement_id for task in tasks_for_hit(hit, snapshot.planner_output)
                          if task.requirement_id not in associated)
    retrieved_missing = missing_evidence_requirement_ids(snapshot.planner_output,
                                                        [hit.evidence for hit in snapshot.parsed_hits], retrieved_map,
                                                        strict=strict)
    return ([item for item in packet_missing if item in retrieved_missing],
            [item for item in packet_missing if item not in retrieved_missing])


def build_validation_snapshot(
    *, user_input: str, planner_output: PlannerOutput, parsed_hits: list[SearchHit],
    current_attempt_retrieval_errors: list[str],
    current_attempt_retrieval_diagnostics: list[RetrievalDiagnostic],
    response_result: AnswerResponse | None, evidence_packet: list[EvidenceRef],
    evidence_requirement_map: dict[str, list[str]] | None = None,
    request_contract: RequestContract | None = None,
    response_kind: str = "draft",
    response_request_id: str | None = None,
    response_contract_revision: int = 0,
    normal_evidence_missing_requirement_ids: list[str] | None = None,
    body_kind: BodyKind | None = None,
    evidence_source: AnswerSource | None = None,
) -> ValidationSnapshot:
    retrieval_required = bool(planner_output.use_retrieval and planner_output.tasks)
    evidence_by_route: dict[str, list[EvidenceRef]] = {"docs": [], "upload": []}
    for hit in parsed_hits:
        evidence_by_route.setdefault(hit.evidence.route, []).append(hit.evidence)
    diagnostics_by_route: dict[str, list[RetrievalDiagnostic]] = {"docs": [], "upload": []}
    for diagnostic in current_attempt_retrieval_diagnostics:
        if diagnostic.route:
            diagnostics_by_route.setdefault(diagnostic.route, []).append(diagnostic)
    return ValidationSnapshot(
        user_input=user_input, planner_output=planner_output, retrieval_required=retrieval_required,
        parsed_hits=parsed_hits, current_attempt_retrieval_errors=current_attempt_retrieval_errors,
        current_attempt_retrieval_diagnostics=current_attempt_retrieval_diagnostics,
        response_result=response_result, evidence_packet=evidence_packet,
        evidence_by_route=evidence_by_route, diagnostics_by_route=diagnostics_by_route,
        required_routes=list(dict.fromkeys(task.route for task in planner_output.tasks)) if retrieval_required else [],
        evidence_requirement_map=evidence_requirement_map or {},
        request_contract=request_contract,
        response_kind=response_kind,
        response_request_id=response_request_id,
        response_contract_revision=response_contract_revision,
        normal_evidence_missing_requirement_ids=normal_evidence_missing_requirement_ids,
        body_kind=body_kind,
        evidence_source=evidence_source,
    )


def collect_validation_snapshot(state: GraphState) -> tuple[ValidationSnapshot, list[str]]:
    local_errors: list[str] = []
    runtime = get_runtime_state(state)
    planner = get_planner_state(state)
    retrieval = get_retrieval_state(state)
    response = get_response_state(state)
    debug = get_debug_state(state)
    retry = get_retry_state(state)
    planner_output = parse_planner_output(planner.output, local_errors)
    parsed_hits: list[SearchHit] = []
    for index, value in enumerate(slice_from_index(retrieval.hit_log, retry.hit_start_index)):
        try:
            parsed_hits.append(value if isinstance(value, SearchHit) else SearchHit.model_validate(value))
        except (TypeError, ValueError) as exc:
            local_errors.append(f"retrieved_hits[{index}]: invalid search hit ({exc})")
    retrieval_errors = [
        str(error) for error in slice_from_index(debug.retrieval_errors, retry.retrieval_error_start_index)
        if str(error).strip()
    ]
    snapshot = build_validation_snapshot(
        user_input=runtime.user_input, planner_output=planner_output, parsed_hits=parsed_hits,
        current_attempt_retrieval_errors=[*retrieval_errors, *local_errors],
        current_attempt_retrieval_diagnostics=[
            item for item in slice_from_index(debug.retrieval_diagnostics, retry.retrieval_diagnostic_start_index)
            if item is not None
        ],
        response_result=response.result, evidence_packet=response.evidence_packet,
        evidence_requirement_map=response.evidence_requirement_map,
        request_contract=runtime.request_contract,
        response_kind=response.kind,
        response_request_id=response.request_id,
        response_contract_revision=response.contract_revision,
        normal_evidence_missing_requirement_ids=response.normal_evidence_missing_requirement_ids,
        body_kind=response.body_kind,
        evidence_source=response.evidence_source,
    )
    return snapshot, local_errors
