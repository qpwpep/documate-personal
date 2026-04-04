from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from ...answer_schema import (
    AgentResponsePayloadModel,
    ClaimItem,
    SynthesisOutput,
    average_claim_confidence,
    clean_grounded_text,
    build_deterministic_grounded_payload,
    build_empty_response_payload,
    render_payload_from_claims,
)
from ...contracts import GraphState, ResponseState
from ...contracts.debug import RetryReason
from ...contracts.routes import route_for_tool
from ..retry import build_followup_from_routes
from .evidence_validator import ValidationAssessment, ValidationSnapshot


def build_response_payload_updates(
    payload: AgentResponsePayloadModel,
    *,
    attempt: int,
) -> GraphState:
    synthesis_output = SynthesisOutput(
        answer=payload.answer,
        claims=payload.claims,
        confidence=payload.confidence,
    )
    return {
        "messages": [AIMessage(content=payload.answer)],
        "response": ResponseState(
            final_answer=payload.answer,
            payload=payload,
            synthesis_output=synthesis_output,
            synthesis_attempt=attempt,
        ),
    }


def build_followup_updates(answer: str, *, attempt: int) -> GraphState:
    return build_response_payload_updates(
        build_empty_response_payload(answer=answer),
        attempt=attempt,
    )


def _claims_cover_required_routes(
    *,
    claims: list[Any],
    snapshot: ValidationSnapshot,
) -> bool:
    required_routes = {
        str(route or "").strip()
        for route in snapshot.required_routes
        if str(route or "").strip()
    }
    if not required_routes:
        return True

    route_by_source_id = {
        str(item.source_id or "").strip(): route_for_tool(str(item.tool or ""))
        for item in snapshot.parsed_evidence
        if str(item.source_id or "").strip()
    }
    covered_routes: set[str] = set()
    for claim in claims:
        for evidence_id in getattr(claim, "evidence_ids", []) or []:
            route = route_by_source_id.get(str(evidence_id or "").strip())
            if route:
                covered_routes.add(route)
    return required_routes.issubset(covered_routes)


def _is_hybrid_retrieval_request(snapshot: ValidationSnapshot) -> bool:
    required_routes = {
        str(route or "").strip()
        for route in snapshot.required_routes
        if str(route or "").strip()
    }
    return "docs" in required_routes and bool(required_routes.intersection({"upload", "local"}))


def _build_route_balanced_hybrid_payload(
    snapshot: ValidationSnapshot,
) -> AgentResponsePayloadModel | None:
    if not _is_hybrid_retrieval_request(snapshot):
        return None

    official_items = [
        item for item in snapshot.parsed_evidence if route_for_tool(str(item.tool or "")) == "docs"
    ]
    local_items = [
        item
        for item in snapshot.parsed_evidence
        if route_for_tool(str(item.tool or "")) in {"upload", "local"}
    ]
    if not official_items or not local_items:
        return None

    official_item = official_items[0]
    local_item = local_items[0]
    official_text = clean_grounded_text(
        official_item.snippet or official_item.title or official_item.url_or_path
    )
    local_text = clean_grounded_text(
        local_item.snippet or local_item.title or local_item.url_or_path
    )
    if not official_text or not local_text:
        return None

    local_route = route_for_tool(str(local_item.tool or ""))
    local_prefix = "반면 업로드 파일에서는" if local_route == "upload" else "반면 로컬 예시에서는"
    claims = [
        ClaimItem(
            text=f"공식 문서 기준: {official_text}",
            evidence_ids=[str(official_item.source_id or "").strip()],
            confidence=official_item.score,
        ),
        ClaimItem(
            text=f"{local_prefix} {local_text}",
            evidence_ids=[str(local_item.source_id or "").strip()],
            confidence=local_item.score,
        ),
    ]
    confidence = average_claim_confidence(claims)
    payload = render_payload_from_claims(
        claims=claims,
        evidence_items=[official_item, local_item],
        confidence=confidence,
    )
    payload.confidence = confidence
    return payload


def apply_validation_outcome(
    *,
    snapshot: ValidationSnapshot,
    assessment: ValidationAssessment,
    attempt: int,
    needs_retry: bool,
) -> GraphState:
    updates: GraphState = {}
    retry_reason: RetryReason | None = assessment.retry_reason
    if retry_reason is None or needs_retry:
        return updates
    hybrid_payload = _build_route_balanced_hybrid_payload(snapshot)

    if assessment.has_grounded_response_payload and snapshot.response_payload is not None:
        updates.update(
            build_response_payload_updates(
                snapshot.response_payload,
                attempt=attempt,
            )
        )
    elif retry_reason == "unsupported_claims" and assessment.valid_claims:
        filtered_confidence = average_claim_confidence(assessment.valid_claims)
        filtered_payload = render_payload_from_claims(
            claims=assessment.valid_claims,
            evidence_items=snapshot.parsed_evidence,
            confidence=filtered_confidence,
        )
        filtered_payload.confidence = filtered_confidence
        if _claims_cover_required_routes(
            claims=assessment.valid_claims,
            snapshot=snapshot,
        ):
            updates.update(
                build_response_payload_updates(
                    filtered_payload,
                    attempt=attempt,
                )
            )
        elif hybrid_payload is not None:
            updates.update(
                build_response_payload_updates(
                    hybrid_payload,
                    attempt=attempt,
                )
            )
        elif snapshot.retrieval_required and snapshot.parsed_evidence:
            grounded_payload = build_deterministic_grounded_payload(
                evidence_items=snapshot.parsed_evidence,
                fallback_answer="",
            )
            updates.update(
                build_response_payload_updates(
                    grounded_payload,
                    attempt=attempt,
                )
            )
        else:
            updates.update(
                build_response_payload_updates(
                    filtered_payload,
                    attempt=attempt,
                )
            )
    elif snapshot.retrieval_required and snapshot.parsed_evidence:
        grounded_payload = hybrid_payload or build_deterministic_grounded_payload(
            evidence_items=snapshot.parsed_evidence,
            fallback_answer="",
        )
        updates.update(
            build_response_payload_updates(
                grounded_payload,
                attempt=attempt,
            )
        )
    else:
        followup_answer = build_followup_from_routes(snapshot.planner_output, retry_reason)
        updates.update(
            build_followup_updates(
                followup_answer,
                attempt=attempt,
            )
        )
    return updates
