from __future__ import annotations

from dataclasses import dataclass
import re

from src.core.answer_schema import AgentResponsePayloadModel, AnswerSection, SynthesisOutput, average_claim_confidence, build_deterministic_grounded_payload, build_empty_response_payload, render_payload_from_claims
from src.core.evidence import EvidenceItem
from src.runtime.nodes.synthesis.budgets import SynthesisBudgetProfile


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(slots=True)
class RenderedSynthesisPayload:
    payload: AgentResponsePayloadModel
    final_answer: str
    synthesis_output: SynthesisOutput


def build_local_fallback_payload(
    *,
    evidence_items: list[EvidenceItem],
    retrieval_required: bool,
    generic_answer: str,
) -> AgentResponsePayloadModel:
    if evidence_items:
        return build_deterministic_grounded_payload(
            evidence_items=evidence_items,
            fallback_answer=generic_answer,
        )
    if retrieval_required:
        return build_empty_response_payload(answer="")
    return build_empty_response_payload(answer=generic_answer)


def _truncate_sentences(text: str, *, limit: int) -> str:
    normalized = " ".join(str(text or "").replace("\r", "\n").split()).strip()
    if not normalized or limit <= 0:
        return normalized
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_PATTERN.split(normalized) if segment.strip()]
    if len(sentences) <= limit:
        return normalized
    return " ".join(sentences[:limit]).strip()


def _clamp_sections(
    sections: list[AnswerSection],
    *,
    max_section_sentences: int | None,
) -> list[AnswerSection]:
    if max_section_sentences is None:
        return sections
    clamped: list[AnswerSection] = []
    for section in sections:
        if section.kind in {"official_docs", "upload_code", "comparison"}:
            sentence_limit = 2 if section.kind == "comparison" else max_section_sentences
            clamped.append(
                section.model_copy(
                    update={
                        "body": _truncate_sentences(
                            section.body,
                            limit=sentence_limit,
                        )
                    }
                )
            )
            continue
        clamped.append(section)
    return clamped


def enforce_synthesis_output_budget(
    *,
    rendered: RenderedSynthesisPayload,
    evidence_items: list[EvidenceItem],
    budget_profile: SynthesisBudgetProfile,
) -> RenderedSynthesisPayload:
    claims = list(rendered.synthesis_output.claims)
    sections = list(rendered.synthesis_output.sections)

    if budget_profile.max_claims is not None:
        claims = claims[: max(0, int(budget_profile.max_claims))]
    sections = _clamp_sections(
        sections,
        max_section_sentences=budget_profile.max_section_sentences,
    )

    if claims == list(rendered.synthesis_output.claims) and sections == list(rendered.synthesis_output.sections):
        return rendered

    synthesis_output = rendered.synthesis_output.model_copy(
        update={
            "claims": claims,
            "sections": sections,
            "confidence": rendered.synthesis_output.confidence or average_claim_confidence(claims),
        }
    )
    return render_synthesis_payload(synthesis_output, evidence_items)


def render_synthesis_payload(
    synthesis_output: SynthesisOutput,
    evidence_items: list[EvidenceItem],
) -> RenderedSynthesisPayload:
    payload_confidence = synthesis_output.confidence
    if payload_confidence is None:
        payload_confidence = average_claim_confidence(synthesis_output.claims)

    if synthesis_output.claims:
        payload = render_payload_from_claims(
            claims=synthesis_output.claims,
            evidence_items=evidence_items,
            confidence=payload_confidence,
            sections=synthesis_output.sections,
        )
        return RenderedSynthesisPayload(
            payload=payload,
            final_answer=payload.answer,
            synthesis_output=synthesis_output,
        )

    fallback_answer = str(synthesis_output.answer or "").strip()
    payload = build_empty_response_payload(
        answer=fallback_answer,
        confidence=payload_confidence,
        sections=synthesis_output.sections,
    )
    return RenderedSynthesisPayload(
        payload=payload,
        final_answer=payload.answer,
        synthesis_output=synthesis_output,
    )
