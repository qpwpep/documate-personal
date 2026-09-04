from __future__ import annotations

import re
from typing import Any

from src.core.answer_schema.fallbacks import build_deterministic_grounded_payload
from src.core.answer_schema.models import AgentResponsePayloadModel, ClaimItem, normalize_confidence
from src.core.answer_schema.rendering import average_claim_confidence, render_payload_from_claims
from src.core.answer_schema.text_cleaning import clean_grounded_text, summarize_grounded_text
from src.core.contracts.routes import route_for_tool
from src.runtime.nodes.validation.evidence_validator import ValidationSnapshot
from src.runtime.nodes.validation.messages_ko import ROUTE_PREFIX_PATTERN, hybrid_docs_prefix, hybrid_limit_sentence, hybrid_upload_prefix, route_prefix


def _ensure_sentence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


def route_by_source_id(snapshot: ValidationSnapshot) -> dict[str, str]:
    return {
        str(item.source_id or "").strip(): route_for_tool(str(item.tool or ""))
        for item in snapshot.parsed_evidence
        if str(item.source_id or "").strip()
    }


def claim_route(
    *,
    claim: ClaimItem,
    snapshot: ValidationSnapshot,
) -> str:
    source_route_map = route_by_source_id(snapshot)
    for evidence_id in claim.evidence_ids:
        route = source_route_map.get(str(evidence_id or "").strip())
        if route:
            return route
    return ""


def claims_for_routes(
    *,
    claims: list[ClaimItem],
    snapshot: ValidationSnapshot,
    routes: set[str],
) -> list[ClaimItem]:
    return [
        claim
        for claim in claims
        if claim_route(claim=claim, snapshot=snapshot) in routes
    ]


def is_hybrid_retrieval_request(snapshot: ValidationSnapshot) -> bool:
    required_routes = {
        str(route or "").strip()
        for route in snapshot.required_routes
        if str(route or "").strip()
    }
    return {"docs", "upload"}.issubset(required_routes)


def claim_from_evidence_item(
    *,
    evidence_item: Any,
    route: str,
) -> ClaimItem | None:
    source_id = str(getattr(evidence_item, "source_id", "") or "").strip()
    if not source_id:
        return None
    grounded_text = (
        clean_grounded_text(getattr(evidence_item, "snippet", "") or "")
        or clean_grounded_text(getattr(evidence_item, "title", "") or "")
        or clean_grounded_text(getattr(evidence_item, "url_or_path", "") or "")
    )
    if not grounded_text:
        return None
    prefix = route_prefix(route)
    if not prefix:
        return None
    return ClaimItem(
        text=f"{prefix} {_ensure_sentence(grounded_text)}",
        evidence_ids=[source_id],
        confidence=normalize_confidence(getattr(evidence_item, "score", None), clamp=True),
    )


def top_evidence_item_for_routes(
    *,
    snapshot: ValidationSnapshot,
    routes: set[str],
) -> Any | None:
    candidates = [
        item
        for item in snapshot.parsed_evidence
        if route_for_tool(str(item.tool or "")) in routes
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: normalize_confidence(getattr(item, "score", None), clamp=True) or 0.0,
        reverse=True,
    )
    for candidate in candidates:
        if summarize_grounded_text(getattr(candidate, "snippet", "") or ""):
            return candidate
        if summarize_grounded_text(getattr(candidate, "title", "") or ""):
            return candidate
    return None


def _normalize_claim_for_route(
    *,
    claim: ClaimItem,
    route: str,
) -> ClaimItem:
    cleaned = clean_grounded_text(claim.text).rstrip(".!?")
    cleaned = re.sub(ROUTE_PREFIX_PATTERN, "", cleaned)
    prefix = route_prefix(route)
    return ClaimItem(
        text=f"{prefix} {cleaned}.",
        evidence_ids=list(claim.evidence_ids),
        confidence=claim.confidence,
    )


def _strip_route_prefix(text: str) -> str:
    return re.sub(
        ROUTE_PREFIX_PATTERN,
        "",
        str(text or "").strip(),
    )


def rewrite_filtered_hybrid_payload(
    *,
    payload: AgentResponsePayloadModel,
    snapshot: ValidationSnapshot,
) -> AgentResponsePayloadModel:
    if not is_hybrid_retrieval_request(snapshot):
        return payload

    docs_claim = next(iter(claims_for_routes(claims=payload.claims, snapshot=snapshot, routes={"docs"})), None)
    upload_claim = next(
        iter(claims_for_routes(claims=payload.claims, snapshot=snapshot, routes={"upload"})),
        None,
    )
    if docs_claim is None:
        official_item = top_evidence_item_for_routes(snapshot=snapshot, routes={"docs"})
        if official_item is not None:
            docs_claim = claim_from_evidence_item(evidence_item=official_item, route="docs")
    else:
        docs_claim = _normalize_claim_for_route(claim=docs_claim, route="docs")
    if upload_claim is None:
        upload_item = top_evidence_item_for_routes(snapshot=snapshot, routes={"upload"})
        if upload_item is not None:
            upload_claim = claim_from_evidence_item(evidence_item=upload_item, route="upload")
    else:
        upload_claim = _normalize_claim_for_route(claim=upload_claim, route="upload")

    if docs_claim is not None:
        cleaned = summarize_grounded_text(_strip_route_prefix(docs_claim.text))
        if cleaned:
            docs_claim = ClaimItem(
                text=f"{hybrid_docs_prefix()} {cleaned}",
                evidence_ids=docs_claim.evidence_ids,
                confidence=docs_claim.confidence,
            )
    if upload_claim is not None:
        cleaned = summarize_grounded_text(_strip_route_prefix(upload_claim.text))
        if cleaned:
            upload_claim = ClaimItem(
                text=f"{hybrid_upload_prefix()} {cleaned}",
                evidence_ids=upload_claim.evidence_ids,
                confidence=upload_claim.confidence,
            )

    rebuilt_claims = [claim for claim in (docs_claim, upload_claim) if claim is not None]
    if not rebuilt_claims:
        return payload

    rebuilt_payload = render_payload_from_claims(
        claims=rebuilt_claims,
        evidence_items=snapshot.parsed_evidence,
        confidence=average_claim_confidence(rebuilt_claims),
    )
    rebuilt_payload.answer = f"{rebuilt_payload.answer} {hybrid_limit_sentence()}".strip()
    rebuilt_payload.confidence = average_claim_confidence(rebuilt_claims)
    return rebuilt_payload


def build_route_balanced_hybrid_payload(
    snapshot: ValidationSnapshot,
) -> AgentResponsePayloadModel | None:
    if not is_hybrid_retrieval_request(snapshot):
        return None

    official_item = top_evidence_item_for_routes(snapshot=snapshot, routes={"docs"})
    upload_item = top_evidence_item_for_routes(snapshot=snapshot, routes={"upload"})
    if official_item is None or upload_item is None:
        return None

    return build_deterministic_grounded_payload(
        evidence_items=[official_item, upload_item],
        max_claims=2,
        fallback_answer="",
    )


__all__ = [
    "build_route_balanced_hybrid_payload",
    "claim_from_evidence_item",
    "claim_route",
    "claims_for_routes",
    "is_hybrid_retrieval_request",
    "rewrite_filtered_hybrid_payload",
    "route_by_source_id",
    "top_evidence_item_for_routes",
]
