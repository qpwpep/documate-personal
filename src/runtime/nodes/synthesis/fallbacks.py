from __future__ import annotations

from src.core.answer_schema import AnswerResponse, build_grounded_response, finalize_answer, text_document
from src.core.evidence import EvidenceRef


def build_synthesis_fallback(
    *, evidence_packet: list[EvidenceRef], retrieval_required: bool, message: str,
) -> AnswerResponse:
    if evidence_packet:
        return build_grounded_response(evidence_packet, message=message)
    notice = (
        "답변 생성에 필요한 근거를 확보하지 못했습니다. 자료나 질문 범위를 확인해 주세요."
        if retrieval_required else message
    )
    return finalize_answer(text_document(notice), [], retrieval_required=retrieval_required)
