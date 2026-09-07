from __future__ import annotations

from src.core.answer_schema import finalize_answer
from src.core.request_contracts import infer_answer_contract, missing_required_content
from src.runtime.nodes.retry import contains_tool_error
from src.runtime.nodes.validation.models import ValidationAssessment, ValidationSnapshot
from src.runtime.nodes.validation.route_policy import route_error_statuses
from src.runtime.nodes.validation.snapshot import detect_missing_route_coverage, detect_missing_requirement_coverage


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
        single_route_task = sum(item.route == route for item in snapshot.planner_output.tasks) == 1
        diagnostics = [d for d in snapshot.current_attempt_retrieval_diagnostics
                       if d.requirement_id == task.requirement_id or (not d.requirement_id and d.route == route and single_route_task)]
        hits = [h for h in snapshot.parsed_hits if h.requirement_id == task.requirement_id
                or (not h.requirement_id and h.evidence.route == route and single_route_task)]
        statuses = route_error_statuses(diagnostics)
        if "error" in statuses or ("unavailable" in statuses and route != "upload"):
            assessment.tool_error_routes.add(route)
            assessment.failed_requirement_ids.add(task.requirement_id)
        elif (not hits or any(d.answerability in {"missing", "partial"} for d in diagnostics)
              or (task.requirement.specified and not any(d.answerability == "covered" for d in diagnostics))):
            assessment.route_failures[route] = "no_evidence"
            assessment.failed_requirement_ids.add(task.requirement_id)
    if contains_tool_error(snapshot.current_attempt_retrieval_errors) and not assessment.tool_error_routes:
        assessment.tool_error_routes = set(snapshot.required_routes)
    if assessment.blocked_missing_upload:
        assessment.retry_reason = "blocked_missing_upload"
        assessment.failed_routes = {"upload"}
    elif assessment.tool_error_routes:
        assessment.retry_reason = "tool_error"
        assessment.failed_routes = set(assessment.tool_error_routes)
    elif assessment.route_failures:
        assessment.retry_reason = "no_evidence"
        assessment.failed_routes = set(assessment.route_failures)
    return assessment


def assess_validation(snapshot: ValidationSnapshot) -> ValidationAssessment:
    assessment = ValidationAssessment()
    if snapshot.response_result is None:
        assessment.retry_reason = "missing_content"
        assessment.missing_content = ["answer"]
        assessment.error_codes = ["VALIDATION_MISSING_CONTENT"]
        return assessment

    # A retrieved hit that was not supplied to synthesis cannot resolve its references.
    result = finalize_answer(
        snapshot.response_result.content, snapshot.evidence_packet,
        retrieval_required=snapshot.retrieval_required or snapshot.response_result.retrieval_required,
        actions=snapshot.response_result.actions, issues=snapshot.response_result.issues,
    )
    assessment.checked_result = result
    for check in result.checks:
        if check.reference_status == "missing" or check.support_status == "unsupported":
            assessment.invalid_unit_paths.add(check.unit_id)
        else:
            assessment.valid_unit_paths.add(check.unit_id)
    if snapshot.retrieval_required:
        assessment.missing_route_coverage = detect_missing_route_coverage(
            required_routes=snapshot.required_routes, result=result,
            evidence_packet=snapshot.evidence_packet, valid_unit_paths=assessment.valid_unit_paths,
        )
        assessment.failed_requirement_ids = set(detect_missing_requirement_coverage(
            snapshot=snapshot, result=result, valid_unit_paths=assessment.valid_unit_paths,
        ))
        assessment.missing_route_coverage = list(dict.fromkeys([
            *assessment.missing_route_coverage,
            *(task.route for task in snapshot.planner_output.tasks if task.requirement_id in assessment.failed_requirement_ids),
        ]))
    assessment.missing_content = missing_required_content(
        infer_answer_contract(snapshot.user_input), result.content,
    )
    if not result.content.blocks:
        assessment.missing_content = list(dict.fromkeys(["answer", *assessment.missing_content]))
    if any(check.reference_status == "missing" for check in result.checks):
        assessment.retry_reason = "unresolved_references"
        assessment.error_codes.append("VALIDATION_UNRESOLVED_REFERENCES")
    elif assessment.invalid_unit_paths:
        assessment.retry_reason = "missing_content"
    elif assessment.missing_route_coverage:
        assessment.retry_reason = "missing_route_coverage"
    elif assessment.missing_content:
        assessment.retry_reason = "missing_content"
    if assessment.retry_reason in {"missing_content", "missing_route_coverage"}:
        assessment.error_codes.append("VALIDATION_MISSING_CONTENT")
    assessment.failed_routes = set(assessment.missing_route_coverage)
    return assessment
