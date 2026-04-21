from __future__ import annotations

from dataclasses import dataclass

from src.core.answer_schema import AgentResponsePayloadModel, ClaimItem, SynthesisOutput, average_claim_confidence, build_deterministic_grounded_payload, build_empty_response_payload, normalize_confidence, render_payload_from_claims
from src.core.evidence import EvidenceItem
from src.runtime.nodes.synthesis.prompt_builder import parse_plain_summary_segments


@dataclass(slots=True)
class RenderedSynthesisPayload:
    payload: AgentResponsePayloadModel
    final_answer: str
    synthesis_output: SynthesisOutput


def build_plain_summary_attach_payload(
    *,
    content: str,
    evidence_items: list[EvidenceItem],
) -> AgentResponsePayloadModel | None:
    if not evidence_items:
        return None

    limited_evidence = evidence_items[:2]
    segments = parse_plain_summary_segments(content, limit=len(limited_evidence))
    if not segments:
        return None

    adopted_pairs = list(zip(segments[: len(limited_evidence)], limited_evidence))
    if not adopted_pairs:
        return None

    claims: list[ClaimItem] = []
    for segment, evidence_item in adopted_pairs:
        source_id = str(evidence_item.source_id or "").strip()
        if not source_id:
            continue
        claims.append(
            ClaimItem(
                text=segment,
                evidence_ids=[source_id],
                confidence=normalize_confidence(evidence_item.score, clamp=True),
            )
        )

    if not claims:
        return None

    confidence = average_claim_confidence(claims)
    payload = render_payload_from_claims(
        claims=claims,
        evidence_items=limited_evidence,
        confidence=confidence,
    )
    payload.confidence = confidence
    return payload


def build_korean_template_summary_payload(
    *,
    content: str,
    evidence_items: list[EvidenceItem],
) -> AgentResponsePayloadModel | None:
    payload = build_plain_summary_attach_payload(content=content, evidence_items=evidence_items)
    if payload is None:
        return None
    rendered_answer = str(content or "").strip()
    if rendered_answer:
        payload.answer = rendered_answer
    return payload


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
