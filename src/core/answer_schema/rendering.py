from __future__ import annotations

import re
from typing import Iterable

from src.core.evidence import EvidenceItem
from src.core.answer_schema.models import AgentResponsePayloadModel, AnswerSection, ClaimItem, is_placeholder_reference_text, normalize_answer_sections


_CITATION_PATTERN = re.compile(r"\s*\[(?:\d+)\]\s*$")


def render_sections_text(
    sections: Iterable[AnswerSection] | None,
) -> str:
    rendered_blocks: list[str] = []
    for section in normalize_answer_sections(sections):
        heading = str(section.heading or "").strip()
        body = str(section.body or "").strip()
        if heading and body:
            rendered_blocks.append(f"{heading}\n{body}")
        else:
            rendered_blocks.append(heading or body)
    return "\n\n".join(block for block in rendered_blocks if block.strip()).strip()


def resolve_answer_text(
    *,
    answer: str = "",
    sections: Iterable[AnswerSection] | None = None,
) -> str:
    rendered_sections = render_sections_text(sections)
    if rendered_sections:
        return rendered_sections
    fallback_answer = str(answer or "").strip()
    if is_placeholder_reference_text(fallback_answer):
        return ""
    return fallback_answer


def build_empty_response_payload(
    *,
    answer: str = "",
    confidence: float | None = None,
    sections: list[AnswerSection] | None = None,
) -> AgentResponsePayloadModel:
    normalized_sections = normalize_answer_sections(sections)
    return AgentResponsePayloadModel(
        answer=resolve_answer_text(answer=answer, sections=normalized_sections),
        claims=[],
        evidence=[],
        confidence=confidence,
        sections=normalized_sections,
    )


def _strip_trailing_citations(text: str) -> str:
    cleaned = str(text or "").strip()
    while cleaned:
        updated = _CITATION_PATTERN.sub("", cleaned)
        if updated == cleaned:
            break
        cleaned = updated.strip()
    return cleaned


def render_payload_from_claims(
    *,
    claims: Iterable[ClaimItem],
    evidence_items: Iterable[EvidenceItem],
    confidence: float | None,
    sections: Iterable[AnswerSection] | None = None,
) -> AgentResponsePayloadModel:
    evidence_by_id = {
        item.source_id: item
        for item in evidence_items
        if isinstance(item, EvidenceItem) and item.source_id
    }
    ordered_claims = [claim for claim in claims if isinstance(claim, ClaimItem)]

    citation_numbers: dict[str, int] = {}
    adopted_evidence: list[EvidenceItem] = []
    renderable_claims: list[tuple[str, tuple[str, ...]]] = []

    for claim in ordered_claims:
        labels: list[str] = []
        for evidence_id in claim.evidence_ids:
            evidence_item = evidence_by_id.get(evidence_id)
            if evidence_item is None:
                continue
            if evidence_id not in citation_numbers:
                citation_numbers[evidence_id] = len(citation_numbers) + 1
                adopted_evidence.append(evidence_item)
            labels.append(f"[{citation_numbers[evidence_id]}]")

        claim_text = _strip_trailing_citations(claim.text)
        renderable_claims.append((claim_text, tuple(labels)))

    rendered_parts: list[str] = []
    for index, (claim_text, labels) in enumerate(renderable_claims):
        next_labels = renderable_claims[index + 1][1] if index + 1 < len(renderable_claims) else ()
        if labels and labels != next_labels:
            rendered_parts.append(f"{claim_text} {' '.join(labels)}")
        else:
            rendered_parts.append(claim_text)

    answer_text = " ".join(part.strip() for part in rendered_parts if part.strip()).strip()
    normalized_sections = normalize_answer_sections(sections)
    return AgentResponsePayloadModel(
        answer=resolve_answer_text(answer=answer_text, sections=normalized_sections),
        claims=ordered_claims,
        evidence=adopted_evidence,
        confidence=confidence,
        sections=normalized_sections,
    )


def filter_claims_by_evidence(
    *,
    claims: Iterable[ClaimItem],
    evidence_items: Iterable[EvidenceItem],
) -> tuple[list[ClaimItem], list[ClaimItem]]:
    evidence_ids = {
        item.source_id
        for item in evidence_items
        if isinstance(item, EvidenceItem) and item.source_id
    }

    valid_claims: list[ClaimItem] = []
    invalid_claims: list[ClaimItem] = []
    for claim in claims:
        if not isinstance(claim, ClaimItem):
            continue
        if not claim.evidence_ids:
            invalid_claims.append(claim)
            continue
        if any(evidence_id not in evidence_ids for evidence_id in claim.evidence_ids):
            invalid_claims.append(claim)
            continue
        valid_claims.append(claim)
    return valid_claims, invalid_claims


def average_claim_confidence(claims: Iterable[ClaimItem]) -> float | None:
    values = [claim.confidence for claim in claims if claim.confidence is not None]
    if not values:
        return None
    return sum(values) / len(values)


__all__ = [
    "average_claim_confidence",
    "build_empty_response_payload",
    "filter_claims_by_evidence",
    "render_payload_from_claims",
    "render_sections_text",
    "resolve_answer_text",
]
