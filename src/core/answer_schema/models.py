from __future__ import annotations

import math
import re
from typing import Iterable

from pydantic import BaseModel, Field, field_validator

from src.core.evidence import EvidenceItem

_PLACEHOLDER_REFERENCE_PATTERN = re.compile(
    r"(?:"
    r"(?:위|아래|앞(?:의)?|다음|상기)\s*(?:코드|예제|내용|자료|문서|결과|설명)\s*(?:를|을)?\s*(?:참고|참조|보세요|확인)"
    r"|(?:see|refer to|as shown in|shown in|use)\s+(?:the\s+)?(?:above|below|previous|following)\s+"
    r"(?:code|example|content|text|result)"
    r"|(?:above|below|previous|following)\s+(?:code|example|content|text|result)"
    r")",
    flags=re.I,
)
_COMPACT_PLACEHOLDER_PHRASES = {
    "위코드참고",
    "위코드참조",
    "아래코드참고",
    "아래코드참조",
    "위예제참고",
    "위내용참고",
    "상기내용참고",
    "seeabovecode",
    "refertoabovecode",
    "aboveexample",
}
_CODE_DETAIL_PATTERN = re.compile(
    r"```|`[^`]+`|"
    r"\b[A-Za-z_][A-Za-z0-9_.]*\s*\(|"
    r"\b[A-Za-z_][A-Za-z0-9_]*\s*=|"
    r"</?[A-Za-z][^>]*>",
    flags=re.M,
)


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip().lower())


def _has_code_or_substantial_detail(text: str) -> bool:
    normalized = " ".join(str(text or "").split())
    if _CODE_DETAIL_PATTERN.search(normalized):
        return True
    return len(normalized) >= 80


def is_placeholder_reference_text(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    compact = _compact_text(normalized)
    has_placeholder_phrase = any(phrase in compact for phrase in _COMPACT_PLACEHOLDER_PHRASES)
    has_placeholder_phrase = has_placeholder_phrase or bool(_PLACEHOLDER_REFERENCE_PATTERN.search(normalized))
    if not has_placeholder_phrase:
        return False
    return not _has_code_or_substantial_detail(normalized)


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
        if not normalized_section.body:
            continue
        if is_placeholder_reference_text(normalized_section.body):
            continue
        normalized_sections.append(normalized_section)
    return normalized_sections


__all__ = [
    "AgentResponsePayloadModel",
    "AnswerSection",
    "ClaimItem",
    "SynthesisOutput",
    "is_placeholder_reference_text",
    "normalize_answer_sections",
    "normalize_confidence",
]
