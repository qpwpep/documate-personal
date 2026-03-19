from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from ...answer_schema import (
    AgentResponsePayloadModel,
    SynthesisOutput,
    average_claim_confidence,
    build_deterministic_grounded_payload,
    build_empty_response_payload,
    render_payload_from_claims,
)
from ...contracts import GraphState, ResponseState
from ...contracts.debug import RetryReason
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
        updates.update(
            build_response_payload_updates(
                filtered_payload,
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
        followup_answer = build_followup_from_routes(snapshot.planner_output, retry_reason)
        updates.update(
            build_followup_updates(
                followup_answer,
                attempt=attempt,
            )
        )
    return updates
