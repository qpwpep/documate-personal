from __future__ import annotations

import time
from typing import Any

from ...answer_schema import SynthesisOutput, build_deterministic_grounded_payload, build_empty_response_payload
from ...contracts import GraphState
from ...latency import elapsed_ms, make_stage_latency_event, make_synthesis_attempt_latency_event
from ..actions import build_action_only_answer, is_action_only_request
from .models import SynthesisContext
from .payload_builder import should_use_deterministic_grounded_direct
from .state import build_synthesis_updates


def maybe_short_circuit_synthesis(
    *,
    state: GraphState,
    debug: Any,
    context: SynthesisContext,
    stage_started: float,
) -> GraphState | None:
    if context.guided_followup:
        payload = build_empty_response_payload(answer=context.guided_followup)
        synthesis_output = SynthesisOutput(answer=context.guided_followup)
        return build_synthesis_updates(
            debug=debug,
            payload=payload,
            synthesis_output=synthesis_output,
            final_answer=context.guided_followup,
            attempt=context.attempt,
            latency_trace=[
                make_stage_latency_event(
                    stage="synthesis",
                    attempt=context.attempt,
                    latency_ms=elapsed_ms(stage_started, time.perf_counter()),
                    status="guided_followup",
                )
            ],
        )

    if is_action_only_request(context.user_input):
        final_answer = build_action_only_answer(
            user_input=context.user_input,
            messages=context.messages,
            slack_target_available=context.slack_target_available,
        )
        payload = build_empty_response_payload(answer=final_answer)
        synthesis_output = SynthesisOutput(answer=final_answer)
        return build_synthesis_updates(
            debug=debug,
            payload=payload,
            synthesis_output=synthesis_output,
            final_answer=final_answer,
            attempt=context.attempt,
            latency_trace=[
                make_stage_latency_event(
                    stage="synthesis",
                    attempt=context.attempt,
                    latency_ms=elapsed_ms(stage_started, time.perf_counter()),
                    status="action_only",
                )
            ],
        )

    if not should_use_deterministic_grounded_direct(
        planner_output=context.planner_output,
        evidence_items=context.primary_evidence_items,
    ):
        return None

    payload = build_deterministic_grounded_payload(
        evidence_items=context.primary_evidence_items,
        fallback_answer="",
    )
    total_ms = elapsed_ms(stage_started, time.perf_counter())
    synthesis_output = SynthesisOutput(
        answer=payload.answer,
        claims=payload.claims,
        confidence=payload.confidence,
    )
    return build_synthesis_updates(
        debug=debug,
        payload=payload,
        synthesis_output=synthesis_output,
        final_answer=payload.answer,
        attempt=context.attempt,
        latency_trace=[
            make_synthesis_attempt_latency_event(
                attempt=context.attempt,
                mode="deterministic_grounded_direct",
                structured_ms=0,
                fallback_ms=None,
                total_ms=total_ms,
            ),
            make_stage_latency_event(
                stage="synthesis",
                attempt=context.attempt,
                latency_ms=total_ms,
                status="deterministic_grounded_direct",
            ),
        ],
        retrieval_errors=context.parse_errors,
        planner_errors=context.planner_parse_errors,
    )
