from __future__ import annotations

from src.core.answer_schema import AnswerResponse, finalize_answer, iter_content_units
from src.core.answer_schema.models import validated_evidence_by_id
from src.core.request_contracts import check_answer_contract
from src.runtime.nodes.validation.models import ValidationAssessment, ValidationSnapshot
from src.runtime.nodes.validation.route_policy import route_error_statuses
from src.runtime.nodes.validation.snapshot import detect_missing_requirement_coverage


def assess_retrieval_quality(snapshot: ValidationSnapshot) -> ValidationAssessment:
    assessment = ValidationAssessment()
    if not snapshot.retrieval_required:
        return assessment
    assessment.blocked_missing_upload = bool(
        "upload" in snapshot.required_routes
        and any(item.status == "unavailable" for item in snapshot.diagnostics_by_route.get("upload", []))
    )
    for task in snapshot.planner_output.tasks:
        route = task.route
        diagnostics = [d for d in snapshot.current_attempt_retrieval_diagnostics
                       if d.requirement_id == task.requirement_id]
        hits = [h for h in snapshot.parsed_hits if h.requirement_id == task.requirement_id]
        statuses = route_error_statuses(diagnostics)
        if route == "upload" and "unavailable" in statuses:
            assessment.failed_requirement_ids.add(task.requirement_id)
        elif "error" in statuses or "unavailable" in statuses:
            assessment.tool_error_routes.add(route)
            assessment.failed_requirement_ids.add(task.requirement_id)
        elif (not hits or any(d.answerability in {"missing", "partial"} for d in diagnostics)
              or (task.requirement.specified and not any(d.answerability == "covered" for d in diagnostics))):
            assessment.route_failures[route] = "no_evidence"
            assessment.failed_requirement_ids.add(task.requirement_id)
    if assessment.tool_error_routes:
        assessment.retry_reason = "tool_error"
        assessment.failed_routes = set(assessment.tool_error_routes)
    elif assessment.route_failures:
        assessment.retry_reason = "no_evidence"
        assessment.failed_routes = set(assessment.route_failures)
    elif assessment.blocked_missing_upload:
        assessment.retry_reason = "blocked_missing_upload"
        assessment.failed_routes = {"upload"}
    return assessment


def assess_validation(snapshot: ValidationSnapshot) -> ValidationAssessment:
    assessment = ValidationAssessment()
    if snapshot.response_result is None:
        assessment.retry_reason = "missing_content"
        assessment.missing_content = ["answer"]
        assessment.error_codes = ["VALIDATION_MISSING_CONTENT"]
        return assessment

    # Validate the recorded revision before applying a genuinely changed packet.
    # Re-finalizing unchanged content would conceal in-memory tampering.
    result = AnswerResponse.model_validate(snapshot.response_result.model_dump(mode="json"))
    packet_by_id = validated_evidence_by_id(snapshot.evidence_packet)
    referenced = {ref for _, unit in iter_content_units(result.content) for ref in unit.refs}
    packet_references = {ref: packet_by_id[ref] for ref in referenced if ref in packet_by_id}
    recorded_references = {citation.evidence.id: citation.evidence for citation in result.citations}
    retrieval_required = snapshot.retrieval_required or result.retrieval_required
    if packet_references != recorded_references or retrieval_required != result.retrieval_required:
        # A retrieved hit outside the actual synthesis packet cannot resolve refs.
        result = finalize_answer(
            result.content, snapshot.evidence_packet, retrieval_required=retrieval_required,
            actions=result.actions, issues=result.issues,
        )
    assessment.checked_result = result
    if snapshot.response_kind in {"clarification", "acknowledgement", "failure"}:
        return assessment
    for check in result.checks:
        if check.reference_status == "missing" or check.support_status == "unsupported":
            assessment.invalid_unit_paths.add(check.unit_id)
        else:
            assessment.valid_unit_paths.add(check.unit_id)
    if snapshot.retrieval_required:
        assessment.failed_requirement_ids = set(detect_missing_requirement_coverage(
            snapshot=snapshot, result=result, valid_unit_paths=assessment.valid_unit_paths,
        ))
        assessment.missing_route_coverage = list(dict.fromkeys([
            task.route for task in snapshot.planner_output.tasks
            if task.requirement_id in assessment.failed_requirement_ids
        ]))
    recoverable_missing = assessment.failed_requirement_ids - set(snapshot.terminal_failed_requirement_ids)
    contract = snapshot.request_contract
    if contract is None or not contract.can_prepare_body():
        assessment.missing_content = ["prepared_body_request"]
    elif (snapshot.response_request_id != contract.request_id
          or snapshot.response_contract_revision != contract.revision):
        assessment.missing_content = ["current_contract_revision"]
    else:
        contract_check = check_answer_contract(snapshot.request_contract.answer, result.content, evidence=snapshot.evidence_packet)
        assessment.missing_content = contract_check.missing_required
        assessment.forbidden_content = contract_check.forbidden_present
    if not result.content.blocks:
        assessment.missing_content = list(dict.fromkeys(["answer", *assessment.missing_content]))
    if any(check.reference_status == "missing" for check in result.checks):
        assessment.retry_reason = "unresolved_references"
        assessment.error_codes.append("VALIDATION_UNRESOLVED_REFERENCES")
    elif assessment.invalid_unit_paths:
        assessment.retry_reason = "missing_content"
    elif recoverable_missing:
        assessment.retry_reason = "missing_route_coverage"
    elif assessment.missing_content or assessment.forbidden_content:
        assessment.retry_reason = "missing_content"
    if assessment.retry_reason in {"missing_content", "missing_route_coverage"}:
        assessment.error_codes.append("VALIDATION_MISSING_CONTENT")
    if (snapshot.terminal_failed_requirement_ids and not result.citations
            and assessment.retry_reason is None):
        assessment.retry_reason = "missing_content"
        assessment.missing_content.append("available_evidence")
    assessment.failed_routes = {task.route for task in snapshot.planner_output.tasks
                                if task.requirement_id in recoverable_missing}
    return assessment
