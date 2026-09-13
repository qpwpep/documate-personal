from __future__ import annotations

from langchain_core.messages import AIMessage

from src.core.answer_schema import (
    AnswerResponse, ResponseIssue, export_answer_text,
    filter_document_units, finalize_answer, text_document,
)
from src.core.contracts import GraphState, ResponseState
from src.core.contracts.provenance import AnswerSource, BodyKind
from src.core.evidence import EvidenceRef
from src.core.request_contracts import check_answer_contract
from src.runtime.nodes.retry import build_followup_from_routes
from src.runtime.nodes.synthesis.fallbacks import build_synthesis_fallback
from src.runtime.nodes.synthesis.evidence_selection import select_evidence_hits
from src.runtime.nodes.validation.models import ValidationAssessment, ValidationSnapshot
from src.runtime.nodes.validation.snapshot import detect_missing_route_coverage, detect_missing_requirement_coverage, detect_packet_coverage_gaps


def build_response_updates(
    result: AnswerResponse, *, attempt: int, evidence_packet: list[EvidenceRef],
    evidence_requirement_map: dict[str, list[str]] | None = None,
    kind: str = "answer",
    request_id: str | None = None,
    contract_revision: int = 0,
    normal_evidence_missing_requirement_ids: list[str] | None = None,
    body_kind: BodyKind | None = None,
    evidence_source: AnswerSource | None = None,
) -> GraphState:
    return {
        "messages": [AIMessage(content=export_answer_text(result))],
        "response": ResponseState(result=result, evidence_packet=evidence_packet, synthesis_attempt=attempt,
                                  evidence_requirement_map=evidence_requirement_map or {}, kind=kind,
                                  normal_evidence_missing_requirement_ids=normal_evidence_missing_requirement_ids,
                                  request_id=request_id, contract_revision=contract_revision,
                                  body_kind=body_kind, evidence_source=evidence_source),
    }


def build_followup_updates(
    answer: str, *, attempt: int, kind: str = "clarification",
    request_id: str | None = None, contract_revision: int = 0,
    body_kind: BodyKind | None = None, evidence_source: AnswerSource | None = None,
) -> GraphState:
    return build_response_updates(
        finalize_answer(text_document(answer), []), attempt=attempt, evidence_packet=[], kind=kind,
        request_id=request_id, contract_revision=contract_revision,
        body_kind=body_kind, evidence_source=evidence_source,
    )


def apply_validation_outcome(
    *, snapshot: ValidationSnapshot, assessment: ValidationAssessment,
    attempt: int, needs_retry: bool,
) -> GraphState:
    if needs_retry:
        return {}
    result = assessment.checked_result
    packet = snapshot.evidence_packet
    contract = snapshot.request_contract
    stamp = {
        "request_id": contract.request_id if contract else None,
        "contract_revision": contract.revision if contract else 0,
        "body_kind": snapshot.body_kind,
        "evidence_source": snapshot.evidence_source,
    }
    if assessment.retry_reason is None and result is not None:
        return build_response_updates(result, attempt=attempt, evidence_packet=packet,
                                      evidence_requirement_map=snapshot.evidence_requirement_map,
                                      normal_evidence_missing_requirement_ids=snapshot.normal_evidence_missing_requirement_ids,
                                      kind=snapshot.response_kind if snapshot.response_kind in {"clarification", "failure"} else "answer",
                                      **stamp)

    if (contract is None or not contract.can_prepare_body()
            or snapshot.response_request_id != contract.request_id
            or snapshot.response_contract_revision != contract.revision):
        return build_followup_updates(
            "요청의 조건을 확정하지 못했습니다. 다시 요청해 주세요.", attempt=attempt, kind="failure",
            **stamp,
        )

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
        contract_check = check_answer_contract(snapshot.request_contract.answer, document, evidence=packet)
        missing_requirements = detect_missing_requirement_coverage(snapshot=snapshot, result=result, valid_unit_paths=valid_paths)
        if document.blocks and not missing_routes and contract_check.valid and not missing_requirements:
            return build_response_updates(result, attempt=attempt, evidence_packet=packet,
                                          evidence_requirement_map=snapshot.evidence_requirement_map,
                                          normal_evidence_missing_requirement_ids=snapshot.normal_evidence_missing_requirement_ids,
                                          **stamp)

    fallback_hits = select_evidence_hits(
        user_input=snapshot.user_input, hits=snapshot.parsed_hits,
        planner_output=snapshot.planner_output,
    )
    if fallback_hits:
        retrieved_missing, packet_omitted = detect_packet_coverage_gaps(snapshot)
        if packet_omitted:
            retained_issues.append(ResponseIssue(
                code="evidence_packet_incomplete",
                message="검색한 원문 중 필요한 내용을 답변 준비 범위에 모두 담지 못했습니다.",
            ))
        if retrieved_missing:
            retained_issues.append(ResponseIssue(
                code="retrieved_evidence_incomplete",
                message="검색한 원문에서 요청에 필요한 근거를 모두 확인하지 못했습니다.",
            ))
        packet = list({hit.evidence.id: hit.evidence for hit in fallback_hits}.values())
        result = build_synthesis_fallback(
            evidence_packet=packet, retrieval_required=True,
            message="요청한 답변을 충분히 구성하지 못해, 확인 가능한 원문 발췌를 제공합니다.",
            request_contract=snapshot.request_contract,
        )
        issues = [*retained_issues, *result.issues, ResponseIssue(
            code="answer_incomplete",
            message=("아래 발췌는 요청 전체에 대한 설명이나 비교가 아닙니다."
                     if result.citations else "요청 전체의 조건을 충족하는 답변을 완성하지 못했습니다."),
        )]
        result = finalize_answer(
            result.content, packet, retrieval_required=result.retrieval_required,
            actions=snapshot.response_result.actions if snapshot.response_result else [],
            issues=issues,
        )
        requirement_map: dict[str, list[str]] = {}
        for hit in fallback_hits:
            if hit.requirement_id:
                requirement_map.setdefault(hit.evidence.id, []).append(hit.requirement_id)
        return build_response_updates(result, attempt=attempt, evidence_packet=packet,
                                      evidence_requirement_map=requirement_map, kind="failure",
                                      normal_evidence_missing_requirement_ids=snapshot.normal_evidence_missing_requirement_ids,
                                      **stamp)

    return build_followup_updates(
        ("요청의 필수·금지 조건을 충족하는 답변을 완성하지 못했습니다. 다시 요청해 주세요."
         if assessment.missing_content or assessment.forbidden_content else
         build_followup_from_routes(snapshot.planner_output, assessment.retry_reason or "missing_content")),
        attempt=attempt, kind="failure", **stamp,
    )
