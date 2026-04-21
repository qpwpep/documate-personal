from __future__ import annotations

import math
from typing import Iterable

from pydantic import BaseModel, Field, field_validator

from src.core.evidence import EvidenceItem


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


class AnswerSection(BaseModel):
    kind: str = Field(min_length=1)
    heading: str = ""
    body: str = ""


class SynthesisOutput(BaseModel):
    answer: str = ""
    claims: list[ClaimItem] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sections: list[AnswerSection] = Field(default_factory=list)


class AgentResponsePayloadModel(BaseModel):
    answer: str = ""
    claims: list[ClaimItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sections: list[AnswerSection] = Field(default_factory=list)


def normalize_answer_sections(
    sections: Iterable[AnswerSection] | None,
) -> list[AnswerSection]:
    normalized_sections: list[AnswerSection] = []
    for section in sections or []:
        if not isinstance(section, AnswerSection):
            continue
        normalized_section = section.model_copy(
            update={
                "kind": str(section.kind or "").strip(),
                "heading": str(section.heading or "").strip(),
                "body": str(section.body or "").strip(),
            }
        )
        if not normalized_section.kind:
            continue
        if not normalized_section.heading and not normalized_section.body:
            continue
        normalized_sections.append(normalized_section)
    return normalized_sections


__all__ = [
    "AgentResponsePayloadModel",
    "AnswerSection",
    "ClaimItem",
    "SynthesisOutput",
    "normalize_answer_sections",
    "normalize_confidence",
]
