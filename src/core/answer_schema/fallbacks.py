from __future__ import annotations

from src.core.evidence import EvidenceRef
from src.core.answer_schema.models import AnswerDocument, CodeBlock, ContentUnit, ParagraphBlock, ResponseIssue
from src.core.answer_schema.rendering import finalize_answer, text_document


def build_grounded_response(evidence: list[EvidenceRef], *, message: str = "답변을 충분히 구성하지 못해 확인 가능한 원문 발췌를 제공합니다."):
    blocks = []
    for item in evidence:
        if not item.excerpt.strip():
            continue
        unit = ContentUnit(text=item.excerpt, basis="excerpt", refs=[item.id])
        if item.element.kind == "code":
            blocks.append(CodeBlock(language=item.element.language or "", content=unit))
        else:
            blocks.append(ParagraphBlock(content=[unit]))
    if not blocks:
        return finalize_answer(text_document("답변에 필요한 근거를 찾지 못했습니다. 자료나 질문 범위를 확인해 주세요."), [])
    return finalize_answer(AnswerDocument(blocks=blocks), evidence, issues=[ResponseIssue(code="source_excerpt_fallback", message=message)])
