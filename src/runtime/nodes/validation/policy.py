from __future__ import annotations

from langchain_core.messages import AIMessage

from src.core.answer_schema import (
    AnswerResponse, ResponseIssue, build_grounded_response, export_answer_text,
    filter_document_units, finalize_answer, text_document,
)
from src.core.contracts import GraphState, ResponseState
from src.core.evidence import EvidenceRef
from src.core.request_contracts import infer_answer_contract, missing_required_content
from src.runtime.nodes.retry import build_followup_from_routes
from src.runtime.nodes.validation.models import ValidationAssessment, ValidationSnapshot
from src.runtime.nodes.validation.snapshot import detect_missing_route_coverage, detect_missing_requirement_coverage


def build_response_updates(
    result: AnswerResponse, *, attempt: int, evidence_packet: list[EvidenceRef],
    evidence_requirement_map: dict[str, list[str]] | None = None,
) -> GraphState:
    return {
        "messages": [AIMessage(content=export_answer_text(result))],
        "response": ResponseState(result=result, evidence_packet=evidence_packet, synthesis_attempt=attempt,
                                  evidence_requirement_map=evidence_requirement_map or {}),
    }


def build_followup_updates(answer: str, *, attempt: int) -> GraphState:
    return build_response_updates(
        finalize_answer(text_document(answer), []), attempt=attempt, evidence_packet=[],
    )


def apply_validation_outcome(
    *, snapshot: ValidationSnapshot, assessment: ValidationAssessment,
    attempt: int, needs_retry: bool,
) -> GraphState:
    if needs_retry:
        return {}
    result = assessment.checked_result
    packet = snapshot.evidence_packet
    if assessment.retry_reason is None and result is not None:
        return build_response_updates(result, attempt=attempt, evidence_packet=packet,
                                      evidence_requirement_map=snapshot.evidence_requirement_map)

    retained_issues = [
        issue for issue in (snapshot.response_result.issues if snapshot.response_result else [])
        if issue.unit_id is None
    ]
    if result is not None and assessment.invalid_unit_paths:
        document = filter_document_units(result.content, assessment.valid_unit_paths)
        result = finalize_answer(
            document, packet, retrieval_required=result.retrieval_required,
            actions=result.actions,
            issues=[*retained_issues, ResponseIssue(
                code="invalid_content_removed",
                message="원문 근거를 연결할 수 없거나 발췌와 일치하지 않는 내용을 제외했습니다.",
            )],
        )
        valid_paths = {check.unit_id for check in result.checks if check.reference_status != "missing"}
        missing_routes = detect_missing_route_coverage(
            required_routes=snapshot.required_routes, result=result,
            evidence_packet=packet, valid_unit_paths=valid_paths,
        ) if snapshot.retrieval_required else []
        missing_content = missing_required_content(
            infer_answer_contract(snapshot.user_input), document,
        )
        missing_requirements = detect_missing_requirement_coverage(snapshot=snapshot, result=result, valid_unit_paths=valid_paths)
        if document.blocks and not missing_routes and not missing_content and not missing_requirements:
            return build_response_updates(result, attempt=attempt, evidence_packet=packet,
                                          evidence_requirement_map=snapshot.evidence_requirement_map)

    if snapshot.parsed_hits:
        packet = list({hit.evidence.id: hit.evidence for hit in snapshot.parsed_hits}.values())
        result = build_grounded_response(
            packet,
            message="요청한 답변을 충분히 구성하지 못해, 확인 가능한 원문 발췌를 제공합니다.",
        )
        issues = [*retained_issues, *result.issues, ResponseIssue(
            code="answer_incomplete",
            message="아래 발췌는 요청 전체에 대한 설명이나 비교가 아닙니다.",
        )]
        result = finalize_answer(
            result.content, packet, retrieval_required=True,
            actions=snapshot.response_result.actions if snapshot.response_result else [],
            issues=issues,
        )
        requirement_map: dict[str, list[str]] = {}
        for hit in snapshot.parsed_hits:
            if hit.requirement_id:
                requirement_map.setdefault(hit.evidence.id, []).append(hit.requirement_id)
        return build_response_updates(result, attempt=attempt, evidence_packet=packet, evidence_requirement_map=requirement_map)

    return build_followup_updates(
        build_followup_from_routes(snapshot.planner_output, assessment.retry_reason or "missing_content"),
        attempt=attempt,
    )
