from __future__ import annotations

import math
import re
from typing import Iterable

from pydantic import BaseModel, Field, field_validator

from .evidence import EvidenceItem


_CITATION_PATTERN = re.compile(r"\s*\[(?:\d+)\]\s*$")
_LEADING_TITLE_PATTERN = re.compile(r"(?i)^title:\s*")
_MARKDOWN_LINK_PATTERN = re.compile(r"!?\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_MARKDOWN_HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s+")
_LEADING_MARKDOWN_PATTERN = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)")
_MARKDOWN_DECORATION_PATTERN = re.compile(r"[*~`]+")
_NAVIGATION_LINE_PATTERNS = (
    re.compile(r"(?i)^(?:api|api reference|documentation|docs?|guide|reference|tutorials?|user guide)$"),
    re.compile(r"(?i)^(?:table of contents|contents|on this page|in this article)$"),
    re.compile(r"(?i)^(?:next|previous|prev|back to top|edit this page|view source|search|skip to content)$"),
    re.compile(r"(?i)^(?:navigation|menu|breadcrumbs?|home)$"),
)
_NAVIGATION_PREFIX_PATTERNS = (
    re.compile(r"(?i)^(?:table of contents|contents|on this page|in this article)\b"),
    re.compile(r"(?i)^(?:next|previous|prev)\s*[:\-]?\s+\S"),
    re.compile(r"(?i)^(?:navigation|menu|breadcrumbs?)\s*[:\-]?\s+\S"),
)
_BREADCRUMB_SPLIT_PATTERN = re.compile(r"\s*(?:[>]|[|]|/|›|»)\s*")
_BREADCRUMB_WORDS = {
    "api",
    "article",
    "back",
    "content",
    "reference",
    "references",
    "doc",
    "docs",
    "documentation",
    "edit",
    "guide",
    "guides",
    "home",
    "in",
    "learn",
    "navigation",
    "next",
    "of",
    "on",
    "overview",
    "page",
    "previous",
    "reference",
    "search",
    "skip",
    "source",
    "this",
    "to",
    "tutorial",
    "tutorials",
    "user",
    "view",
}


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
    if not cleaned:
        return ""

    filtered_lines: list[str] = []
    for raw_line in cleaned.replace("\r", "\n").split("\n"):
        is_markdown_heading = _MARKDOWN_HEADING_PATTERN.match(raw_line) is not None
        is_markdown_link_only = _MARKDOWN_LINK_PATTERN.fullmatch(raw_line.strip()) is not None
        line = _MARKDOWN_IMAGE_PATTERN.sub(" ", raw_line)
        line = _MARKDOWN_LINK_PATTERN.sub(r"\1", line)
        line = _HTML_TAG_PATTERN.sub(" ", line)
        line = _LEADING_MARKDOWN_PATTERN.sub("", line).strip()
        line = _MARKDOWN_DECORATION_PATTERN.sub(" ", line)
        line = re.sub(r"\s+", " ", line).strip(" -|:")
        if not line:
            continue
        if is_markdown_heading and len(line.split()) <= 8:
            continue
        if is_markdown_link_only and len(line.split()) <= 8:
            continue
        if _looks_like_navigation_line(line):
            continue
        filtered_lines.append(line)

    cleaned = " ".join(filtered_lines)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+\.\.\.\s*$", "", cleaned)
    return cleaned.strip()


def _looks_like_navigation_line(line: str) -> bool:
    if any(pattern.fullmatch(line) for pattern in _NAVIGATION_LINE_PATTERNS):
        return True
    if any(pattern.match(line) for pattern in _NAVIGATION_PREFIX_PATTERNS):
        return True

    line_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]*", line)
    }
    if line_words and len(line_words) <= 8 and line_words.issubset(_BREADCRUMB_WORDS):
        return True

    breadcrumb_segments = [
        segment.strip()
        for segment in _BREADCRUMB_SPLIT_PATTERN.split(line)
        if segment.strip()
    ]
    if len(breadcrumb_segments) <= 1:
        return False
    normalized_segments = [re.sub(r"\s+", " ", segment).strip().lower() for segment in breadcrumb_segments]
    if any(len(segment.split()) > 4 for segment in normalized_segments):
        return False
    if normalized_segments[0] == "home":
        return True
    if any(segment in _BREADCRUMB_WORDS for segment in normalized_segments[:-1]):
        return True
    if all(_looks_like_navigation_line(segment) for segment in breadcrumb_segments):
        return True

    breadcrumb_words = {
        word.lower()
        for segment in breadcrumb_segments
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]*", segment)
    }
    return bool(breadcrumb_words) and breadcrumb_words.issubset(_BREADCRUMB_WORDS)


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
