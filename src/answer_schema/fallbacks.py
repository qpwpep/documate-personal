from __future__ import annotations

from typing import Iterable

from ..evidence import EvidenceItem
from .models import AgentResponsePayloadModel, ClaimItem, normalize_confidence
from .rendering import (
    average_claim_confidence,
    build_empty_response_payload,
    render_payload_from_claims,
)
from .text_cleaning import clean_grounded_text, summarize_grounded_text


def _ensure_sentence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


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

    local_limit = (
        "업로드 파일 1건만"
        if str(local_item.tool or "").strip().lower() == "upload_search"
        else "로컬 자료 1건만"
    )
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
):
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


__all__ = [
    "build_deterministic_grounded_payload",
]
