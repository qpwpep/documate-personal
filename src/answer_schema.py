from __future__ import annotations

import math
import re
from typing import Iterable

from pydantic import BaseModel, Field, field_validator

from .evidence import EvidenceItem


_CITATION_PATTERN = re.compile(r"\s*\[(?:\d+)\]\s*$")
_LEADING_TITLE_PATTERN = re.compile(r"(?i)^title:\s*")


def normalize_confidence(value: object, *, clamp: bool = False) -> float | None:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized):
        return None
    if clamp:
        return max(0.0, min(1.0, normalized))
    if 0.0 <= normalized <= 1.0:
        return normalized
    return None


class ClaimItem(BaseModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("claim text must not be empty")
        return cleaned

    @field_validator("evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            evidence_id = str(item or "").strip()
            if evidence_id and evidence_id not in normalized:
                normalized.append(evidence_id)
        return normalized


class SynthesisOutput(BaseModel):
    answer: str = ""
    claims: list[ClaimItem] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AgentResponsePayloadModel(BaseModel):
    answer: str = ""
    claims: list[ClaimItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


def build_empty_response_payload(
    *,
    answer: str = "",
    confidence: float | None = None,
) -> AgentResponsePayloadModel:
    return AgentResponsePayloadModel(
        answer=str(answer or "").strip(),
        claims=[],
        evidence=[],
        confidence=confidence,
    )


def clean_grounded_text(text: str) -> str:
    cleaned = _LEADING_TITLE_PATTERN.sub("", str(text or "").strip())
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+\.\.\.\s*$", "", cleaned)
    return cleaned.strip()


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
) -> AgentResponsePayloadModel:
    evidence_by_id = {
        item.source_id: item
        for item in evidence_items
        if isinstance(item, EvidenceItem) and item.source_id
    }
    ordered_claims = [claim for claim in claims if isinstance(claim, ClaimItem)]

    citation_numbers: dict[str, int] = {}
    adopted_evidence: list[EvidenceItem] = []
    rendered_parts: list[str] = []

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
        if labels:
            rendered_parts.append(f"{claim_text} {' '.join(labels)}")
        else:
            rendered_parts.append(claim_text)

    answer_text = " ".join(part.strip() for part in rendered_parts if part.strip()).strip()
    return AgentResponsePayloadModel(
        answer=answer_text,
        claims=ordered_claims,
        evidence=adopted_evidence,
        confidence=confidence,
    )


def build_deterministic_grounded_payload(
    *,
    evidence_items: Iterable[EvidenceItem],
    max_claims: int = 2,
    fallback_answer: str = "",
) -> AgentResponsePayloadModel:
    grounded_claims: list[ClaimItem] = []
    normalized_evidence = [
        item for item in evidence_items if isinstance(item, EvidenceItem)
    ]

    for item in normalized_evidence[:max_claims]:
        source_id = str(item.source_id or "").strip()
        if not source_id:
            continue

        fallback_text = (
            clean_grounded_text(item.snippet or "")
            or clean_grounded_text(item.title or "")
            or clean_grounded_text(item.url_or_path or "")
        )
        if not fallback_text:
            continue

        grounded_claims.append(
            ClaimItem(
                text=fallback_text,
                evidence_ids=[source_id],
                confidence=normalize_confidence(item.score, clamp=True),
            )
        )

    if not grounded_claims:
        return build_empty_response_payload(answer=fallback_answer)

    confidence = average_claim_confidence(grounded_claims)
    payload = render_payload_from_claims(
        claims=grounded_claims,
        evidence_items=normalized_evidence,
        confidence=confidence,
    )
    payload.confidence = confidence
    return payload


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
