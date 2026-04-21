from __future__ import annotations

from langchain_core.messages import AIMessage

from src.core.answer_schema.fallbacks import build_deterministic_grounded_payload
from src.core.answer_schema.models import AgentResponsePayloadModel, SynthesisOutput
from src.core.answer_schema.rendering import average_claim_confidence, build_empty_response_payload, render_payload_from_claims
from src.core.contracts import GraphState, ResponseState
from src.core.contracts.debug import RetryReason
from src.runtime.nodes.retry import build_followup_from_routes
from src.runtime.nodes.validation.evidence_validator import ValidationAssessment, ValidationSnapshot
from src.runtime.nodes.validation.hybrid_rewrite import build_route_balanced_hybrid_payload, claims_for_routes, is_hybrid_retrieval_request, rewrite_filtered_hybrid_payload
from src.runtime.nodes.validation.repair import repair_required_sections


def build_response_payload_updates(
    payload: AgentResponsePayloadModel,
    *,
    attempt: int,
) -> GraphState:
    synthesis_output = SynthesisOutput(
        answer=payload.answer,
        claims=payload.claims,
        confidence=payload.confidence,
        sections=payload.sections,
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

    next_payload: AgentResponsePayloadModel | None = None
    if assessment.has_grounded_response_payload and snapshot.response_payload is not None:
        next_payload = snapshot.response_payload.model_copy(deep=True)
    elif assessment.valid_claims:
        filtered_confidence = average_claim_confidence(assessment.valid_claims)
        next_payload = render_payload_from_claims(
            claims=assessment.valid_claims,
            evidence_items=snapshot.parsed_evidence,
            confidence=filtered_confidence,
        )
        next_payload.confidence = filtered_confidence
        if snapshot.response_payload is not None and snapshot.response_payload.sections:
            next_payload = next_payload.model_copy(update={"sections": snapshot.response_payload.sections})
    elif retry_reason == "missing_sections" and snapshot.response_payload is not None:
        next_payload = snapshot.response_payload.model_copy(deep=True)
    elif snapshot.retrieval_required and snapshot.parsed_evidence:
        docs_valid_claims = claims_for_routes(
            claims=assessment.valid_claims,
            snapshot=snapshot,
            routes={"docs"},
        )
        local_valid_claims = claims_for_routes(
            claims=assessment.valid_claims,
            snapshot=snapshot,
            routes={"upload", "local"},
        )
        next_payload = None
        if not docs_valid_claims and not local_valid_claims:
            next_payload = build_route_balanced_hybrid_payload(snapshot)
        next_payload = next_payload or build_deterministic_grounded_payload(
            evidence_items=snapshot.parsed_evidence,
            fallback_answer="",
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

    if next_payload is None:
        return updates

    if snapshot.response_payload is not None and snapshot.response_payload.sections and not next_payload.sections:
        next_payload = next_payload.model_copy(update={"sections": snapshot.response_payload.sections})

    if is_hybrid_retrieval_request(snapshot) and (
        retry_reason in {"unsupported_claims", "missing_route_coverage"}
        or bool(assessment.missing_route_coverage)
    ):
        next_payload = rewrite_filtered_hybrid_payload(
            payload=next_payload,
            snapshot=snapshot,
        )

    next_payload = repair_required_sections(
        payload=next_payload,
        snapshot=snapshot,
    )
    updates.update(
        build_response_payload_updates(
            next_payload,
            attempt=attempt,
        )
    )
    return updates


__all__ = [
    "apply_validation_outcome",
    "build_followup_updates",
    "build_response_payload_updates",
]
