from __future__ import annotations

from src.core.answer_schema import AnswerResponse, ResponseIssue, build_grounded_response, finalize_answer, text_document
from src.core.evidence import EvidenceRef
from src.core.request_contracts import RequestContract, check_answer_contract


def build_synthesis_fallback(
    *, evidence_packet: list[EvidenceRef], retrieval_required: bool, message: str,
    request_contract: RequestContract,
) -> AnswerResponse:
    if evidence_packet:
        result = build_grounded_response(evidence_packet, message=message)
        check = check_answer_contract(request_contract.answer, result.content, evidence=evidence_packet)
        if check.valid and not check.unchecked_semantic:
            return result
        message = "요청의 필수·금지 조건을 충족하는 답변을 완성하지 못했습니다. 다시 요청해 주세요."
    notice = message if evidence_packet else (
        "답변 생성에 필요한 근거를 확보하지 못했습니다. 자료나 질문 범위를 확인해 주세요."
        if retrieval_required else "답변을 완성하지 못했습니다. 다시 요청해 주세요."
    )
    return finalize_answer(text_document(notice), [], issues=[ResponseIssue(
        code="answer_incomplete", message="요청 조건을 충족한 전달 본문을 만들지 못했습니다.",
    )])
