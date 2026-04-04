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
_DOC_SECTION_HEADING_PATTERN = re.compile(
    r"(?i)^(?:parameters?|returns?|examples?|notes?|see also|references?|attributes?|methods?)$"
)
_DOC_REFERENCE_TITLE_PATTERN = re.compile(
    r"(?i)\b(?:documentation|docs?|reference|api reference|user guide)\b"
)
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
_NAVIGATION_EMBEDDED_PATTERNS = (
    re.compile(r"(?i)\bskip to content\b"),
    re.compile(r"(?i)\bon this page\b"),
    re.compile(r"(?i)\btable of contents\b"),
    re.compile(r"(?i)\bedit this page\b"),
    re.compile(r"(?i)\bview source\b"),
)
_BREADCRUMB_SPLIT_PATTERN = re.compile(r"\s*(?:[>]|[|]|/|→)\s*")
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
_DOC_SECTION_WORDS = {
    "parameters",
    "parameter",
    "returns",
    "return",
    "examples",
    "example",
    "notes",
    "note",
    "references",
    "reference",
    "attributes",
    "attribute",
    "methods",
    "method",
    "see",
    "also",
}
_PLAIN_LANGUAGE_SIGNATURE_PREFIXES = {
    "allow",
    "allows",
    "call",
    "calls",
    "create",
    "creates",
    "join",
    "joins",
    "pass",
    "passes",
    "return",
    "returns",
    "set",
    "sets",
    "split",
    "splits",
    "use",
    "uses",
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


def _looks_like_navigation_line(line: str) -> bool:
    if any(pattern.fullmatch(line) for pattern in _NAVIGATION_LINE_PATTERNS):
        return True
    if any(pattern.match(line) for pattern in _NAVIGATION_PREFIX_PATTERNS):
        return True

    line_words = {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9-]*", line)}
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
    return False


def _normalize_doc_line(line: str) -> str:
    normalized = str(line or "").strip()
    if not normalized:
        return ""
    normalized = re.sub(r"\\([_*#`])", r"\1", normalized)
    normalized = normalized.replace("\\", "")
    normalized = re.sub(r"\s*[#]+\s*", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" -|:")


def _looks_like_signature_line(line: str) -> bool:
    if len(line) > 160 or "(" not in line:
        return False
    prefix, _, suffix = line.partition("(")
    identifier = prefix.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{1,}", identifier):
        return False
    inner = suffix.rsplit(")", 1)[0] if ")" in suffix else suffix
    return bool(inner.strip()) and any(marker in inner for marker in (",", "=", "*", "[", "]"))


def _looks_like_signature_fragment(line: str) -> bool:
    normalized = _normalize_doc_line(line)
    if len(normalized) > 220 or "(" not in normalized:
        return False
    prefix, _, suffix = normalized.partition("(")
    prefix_tokens = [token for token in re.sub(r"[#:.]", " ", prefix).split() if token]
    if not prefix_tokens or len(prefix_tokens) > 6:
        return False
    if prefix_tokens[0].lower() in _PLAIN_LANGUAGE_SIGNATURE_PREFIXES:
        return False
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", token) for token in prefix_tokens):
        return False
    inner = suffix.rsplit(")", 1)[0] if ")" in suffix else suffix
    return bool(inner.strip()) and any(marker in inner for marker in (",", "=", "*", "[", "]"))


def _looks_like_title_only_line(line: str) -> bool:
    if len(line) > 60 or any(punct in line for punct in ".!?"):
        return False
    if "(" in line or ")" in line:
        return False
    words = line.split()
    if not 1 <= len(words) <= 4:
        return False
    return all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", word) for word in words)


def _looks_like_identifier_only_fragment(line: str) -> bool:
    stripped = str(line or "").strip().rstrip(".!?")
    words = stripped.split()
    if not words or len(words) > 3:
        return False
    return all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", word) for word in words)


def _looks_like_section_listing(line: str) -> bool:
    section_words = [word.lower() for word in re.findall(r"[A-Za-z]+", line)]
    return bool(section_words) and 2 <= len(section_words) <= 8 and all(
        word in _DOC_SECTION_WORDS for word in section_words
    )


def _looks_like_doc_chrome_line(line: str) -> bool:
    normalized = _normalize_doc_line(line)
    if not normalized:
        return False
    if _DOC_SECTION_HEADING_PATTERN.fullmatch(normalized):
        return True
    if _looks_like_section_listing(normalized):
        return True
    if _looks_like_signature_line(normalized):
        return True
    if _looks_like_signature_fragment(normalized):
        return True
    if len(normalized.split()) <= 12 and _DOC_REFERENCE_TITLE_PATTERN.search(normalized):
        return True
    return _looks_like_title_only_line(normalized)


def clean_grounded_text(text: str) -> str:
    cleaned = _LEADING_TITLE_PATTERN.sub("", str(text or "").strip())
    if not cleaned:
        return ""

    filtered_lines: list[str] = []
    for raw_line in cleaned.replace("\r", "\n").split("\n"):
        is_markdown_heading = _MARKDOWN_HEADING_PATTERN.match(raw_line) is not None
        is_markdown_link_only = _MARKDOWN_LINK_PATTERN.fullmatch(raw_line.strip()) is not None
        line = _MARKDOWN_IMAGE_PATTERN.sub(" ", raw_line)
        for pattern in _NAVIGATION_EMBEDDED_PATTERNS:
            line = pattern.sub(" ", line)
        line = _MARKDOWN_LINK_PATTERN.sub(r"\1", line)
        line = _HTML_TAG_PATTERN.sub(" ", line)
        line = _LEADING_MARKDOWN_PATTERN.sub("", line).strip()
        line = _MARKDOWN_DECORATION_PATTERN.sub(" ", line)
        line = _normalize_doc_line(line)
        if not line:
            continue
        if is_markdown_heading and len(line.split()) <= 8:
            continue
        if is_markdown_link_only and len(line.split()) <= 8:
            continue
        if _looks_like_navigation_line(line) or _looks_like_doc_chrome_line(line):
            continue
        if not filtered_lines and _looks_like_title_only_line(line):
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{1,}\([^)]*\)", line):
            continue
        filtered_lines.append(line)

    cleaned = " ".join(filtered_lines)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+\.\.\.\s*$", "", cleaned)
    if _looks_like_doc_chrome_line(cleaned):
        return ""
    return cleaned.strip()


def summarize_grounded_text(text: str, *, max_chars: int = 220) -> str:
    cleaned = clean_grounded_text(text)
    if not cleaned:
        return ""

    first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    summary = first_sentence or cleaned
    if len(summary) > max_chars:
        summary = (summary[:max_chars].rsplit(" ", 1)[0] or summary[:max_chars]).rstrip(" ,;:")
    if _looks_like_doc_chrome_line(summary) or _looks_like_signature_fragment(summary):
        return ""
    if _looks_like_identifier_only_fragment(summary):
        return ""
    return _ensure_sentence(summary)


def _strip_trailing_citations(text: str) -> str:
    cleaned = str(text or "").strip()
    while cleaned:
        updated = _CITATION_PATTERN.sub("", cleaned)
        if updated == cleaned:
            break
        cleaned = updated.strip()
    return cleaned


def _ensure_sentence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


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


def _local_comparison_prefix(item: EvidenceItem) -> str:
    tool_name = str(item.tool or "").strip().lower()
    if tool_name == "upload_search":
        return "업로드 파일에서는"
    return "로컬 자료에서는"


def _build_hybrid_fallback_payload(
    *,
    evidence_items: list[EvidenceItem],
) -> AgentResponsePayloadModel | None:
    official_item = next(
        (item for item in evidence_items if str(item.kind or "").strip().lower() == "official"),
        None,
    )
    local_item = next(
        (item for item in evidence_items if str(item.kind or "").strip().lower() == "local"),
        None,
    )
    if official_item is None or local_item is None:
        return None

    official_text = (
        summarize_grounded_text(official_item.snippet or "")
        or summarize_grounded_text(official_item.title or "")
        or summarize_grounded_text(official_item.url_or_path or "")
    )
    local_text = (
        summarize_grounded_text(local_item.snippet or "")
        or summarize_grounded_text(local_item.title or "")
        or summarize_grounded_text(local_item.url_or_path or "")
    )
    if not official_text or not local_text:
        return None

    official_source_id = str(official_item.source_id or "").strip()
    local_source_id = str(local_item.source_id or "").strip()
    if not official_source_id or not local_source_id:
        return None

    local_limit = "업로드 파일 1건만" if str(local_item.tool or "").strip().lower() == "upload_search" else "로컬 자료 1건만"
    claims = [
        ClaimItem(
            text=f"공식 문서 기준으로는 {official_text}",
            evidence_ids=[official_source_id],
            confidence=normalize_confidence(official_item.score, clamp=True),
        ),
        ClaimItem(
            text=f"{_local_comparison_prefix(local_item)} {local_text}",
            evidence_ids=[local_source_id],
            confidence=normalize_confidence(local_item.score, clamp=True),
        ),
    ]
    confidence = average_claim_confidence(claims)
    payload = render_payload_from_claims(
        claims=claims,
        evidence_items=[official_item, local_item],
        confidence=confidence,
    )
    payload.answer = (
        f"{payload.answer} 근거는 공식 문서 1건과 {local_limit} 반영했습니다."
    ).strip()
    payload.confidence = confidence
    return payload


def build_deterministic_grounded_payload(
    *,
    evidence_items: Iterable[EvidenceItem],
    max_claims: int = 2,
    fallback_answer: str = "",
) -> AgentResponsePayloadModel:
    grounded_claims: list[ClaimItem] = []
    normalized_evidence = [item for item in evidence_items if isinstance(item, EvidenceItem)]
    evidence_kinds = {
        str(item.kind or "").strip().lower()
        for item in normalized_evidence
    }
    is_hybrid_grounded_payload = "official" in evidence_kinds and "local" in evidence_kinds

    if is_hybrid_grounded_payload:
        hybrid_payload = _build_hybrid_fallback_payload(
            evidence_items=normalized_evidence[: max(2, max_claims)],
        )
        if hybrid_payload is not None:
            return hybrid_payload

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
                text=_ensure_sentence(fallback_text),
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


AgentResponsePayload = AgentResponsePayloadModel
