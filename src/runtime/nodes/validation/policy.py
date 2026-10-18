from __future__ import annotations

from langchain_core.messages import AIMessage

from src.core.answer_schema import (
    AnswerResponse, ResponseIssue, export_answer_text,
    filter_document_units, finalize_answer, text_document,
)
from src.core.contracts import GraphState, ResponseState
from src.core.contracts.provenance import AnswerSource, BodyKind
from src.core.evidence import EvidenceRef
from src.core.llm_errors import ExecutionProblem, make_problem
from src.core.request_contracts import check_answer_contract
from src.runtime.nodes.retry import build_followup_from_routes, retrieval_subject
from src.runtime.nodes.synthesis.fallbacks import build_synthesis_fallback
from src.runtime.nodes.synthesis.evidence_selection import select_evidence_hits
from src.runtime.nodes.validation.models import ValidationAssessment, ValidationSnapshot
from src.runtime.nodes.validation.snapshot import detect_missing_requirement_coverage, detect_packet_coverage_gaps


def build_response_updates(
    result: AnswerResponse, *, attempt: int, evidence_packet: list[EvidenceRef],
    evidence_requirement_map: dict[str, list[str]] | None = None,
    kind: str = "answer",
    request_id: str | None = None,
    contract_revision: int = 0,
    normal_evidence_missing_requirement_ids: list[str] | None = None,
    body_kind: BodyKind | None = None,
    evidence_source: AnswerSource | None = None,
    problem: ExecutionProblem | None = None,
) -> GraphState:
    return {
        "messages": [AIMessage(content=export_answer_text(result))],
        "response": ResponseState(result=result, evidence_packet=evidence_packet, synthesis_attempt=attempt,
                                  evidence_requirement_map=evidence_requirement_map or {}, kind=kind,
                                  normal_evidence_missing_requirement_ids=normal_evidence_missing_requirement_ids or [],
                                  request_id=request_id, contract_revision=contract_revision,
                                  body_kind=body_kind, evidence_source=evidence_source, problem=problem),
    }


def build_followup_updates(
    answer: str, *, attempt: int, kind: str = "clarification",
    request_id: str | None = None, contract_revision: int = 0,
    body_kind: BodyKind | None = None, evidence_source: AnswerSource | None = None,
    problem: ExecutionProblem | None = None,
) -> GraphState:
    return build_response_updates(
        finalize_answer(text_document(answer), []), attempt=attempt, evidence_packet=[], kind=kind,
        request_id=request_id, contract_revision=contract_revision,
        body_kind=body_kind, evidence_source=evidence_source,
        problem=problem,
    )


def _retrieval_gap_message(snapshot: ValidationSnapshot) -> str:
    notices = []
    failed = set(snapshot.terminal_failed_requirement_ids)
    for task in snapshot.planner_output.tasks:
        if task.requirement_id not in failed:
            continue
        requirement = task.requirement
        source = "공식 문서" if task.route == "docs" else "업로드 파일"
        target = retrieval_subject(task)
        details = []
        if requirement.version:
            details.append(f"버전 {requirement.version}")
        if requirement.file_ids:
            names = [snapshot.upload_file_names.get(file_id, file_id) for file_id in requirement.file_ids]
            details.append("파일 " + ", ".join(names))
        if requirement.aspects:
            details.append("확인 항목 " + ", ".join(requirement.aspects))
        if details:
            target += " (" + "; ".join(details) + ")"
        notices.append(f"{source}에서 {target}에 필요한 근거를 충분히 확인하지 못했습니다.")
    return " ".join(notices)


def add_retrieval_gap_issues(snapshot: ValidationSnapshot, result: AnswerResponse) -> AnswerResponse:
    """Keep the checked body and its original failure while exposing source gaps."""
    if not snapshot.terminal_failed_requirement_ids:
        return result
    issues = [issue for issue in result.issues if issue.code != "retrieved_evidence_incomplete"]
    issues.append(ResponseIssue(code="retrieved_evidence_incomplete", message=_retrieval_gap_message(snapshot)))
    if not any(issue.code == "answer_incomplete" for issue in issues):
        issues.append(ResponseIssue(
            code="answer_incomplete",
            message="확인한 근거로 답변을 작성했으며, 요청 전체는 완료하지 못했습니다.",
        ))
    # Content, source ranges, checks and the content hash stay exactly as checked.
    return result.model_copy(update={"issues": issues})


def apply_validation_outcome(
    *, snapshot: ValidationSnapshot, assessment: ValidationAssessment,
    attempt: int, needs_retry: bool, allow_filtered_answer: bool = True,
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
    def checked_response_updates(checked: AnswerResponse) -> GraphState:
        kind = snapshot.response_kind if snapshot.response_kind in {"clarification", "acknowledgement", "failure"} else "answer"
        problem = None
        if snapshot.terminal_failed_requirement_ids:
            checked = add_retrieval_gap_issues(snapshot, checked)
            kind = "failure"
            problem = make_problem(
                "provider_unavailable" if snapshot.terminal_retrieval_reason == "tool_error" else "evidence_insufficient",
                "retrieval",
            )
        return build_response_updates(
            checked, attempt=attempt, evidence_packet=packet,
            evidence_requirement_map=snapshot.evidence_requirement_map,
            normal_evidence_missing_requirement_ids=snapshot.normal_evidence_missing_requirement_ids,
            kind=kind, problem=problem, **stamp,
        )

    if assessment.retry_reason is None and result is not None:
        return checked_response_updates(result)

    if (contract is None or not contract.can_prepare_body()
            or snapshot.response_request_id != contract.request_id
            or snapshot.response_contract_revision != contract.revision):
        return build_followup_updates(
            "요청의 조건을 확정하지 못했습니다. 다시 요청해 주세요.", attempt=attempt, kind="failure",
            problem=make_problem("internal_error", "validation"),
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
                message="근거로 충분히 뒷받침되지 않거나 원문과 일치하지 않는 내용을 제외했습니다.",
            )],
        )
        valid_paths = {check.unit_id for check in result.checks if check.reference_status != "missing"}
        contract_check = check_answer_contract(snapshot.request_contract.answer, document, evidence=packet)
        missing_requirements = detect_missing_requirement_coverage(snapshot=snapshot, result=result, valid_unit_paths=valid_paths)
        recoverable_missing = set(missing_requirements) - set(snapshot.terminal_failed_requirement_ids)
        # Terminal source gaps permit a partial body only while available evidence remains cited.
        has_required_citations = not snapshot.terminal_failed_requirement_ids or bool(result.citations)
        if (allow_filtered_answer and document.blocks and contract_check.valid
                and not recoverable_missing and has_required_citations):
            return checked_response_updates(result)

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
        result = add_retrieval_gap_issues(snapshot, result)
        requirement_map: dict[str, list[str]] = {}
        for hit in fallback_hits:
            if hit.requirement_id:
                requirement_map.setdefault(hit.evidence.id, []).append(hit.requirement_id)
        return build_response_updates(result, attempt=attempt, evidence_packet=packet,
                                      evidence_requirement_map=requirement_map, kind="failure",
                                      problem=make_problem("evidence_insufficient", "retrieval") if retrieved_missing else
                                              make_problem("model_output_invalid", "validation"),
                                      normal_evidence_missing_requirement_ids=snapshot.normal_evidence_missing_requirement_ids,
                                      **stamp)

    return build_followup_updates(
        ("요청의 필수·금지 조건을 충족하는 답변을 완성하지 못했습니다. 다시 요청해 주세요."
         if assessment.missing_content or assessment.forbidden_content else
         build_followup_from_routes(snapshot.planner_output, assessment.retry_reason or "missing_content")),
        attempt=attempt, kind="failure", problem=make_problem("model_output_invalid", "validation"), **stamp,
    )
