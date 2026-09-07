from __future__ import annotations

import time
from typing import Any

from src.core.answer_schema import build_grounded_response, finalize_answer, text_document
from src.core.contracts import GraphState
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.latency import elapsed_ms, make_stage_latency_event
from src.core.prompts import needs_slack
from src.runtime.nodes.actions.policy import is_action_only_request
from src.runtime.nodes.synthesis.models import SynthesisContext
from src.runtime.nodes.synthesis.request_intent import is_explicit_source_extraction
from src.runtime.nodes.synthesis.state import build_synthesis_updates


def maybe_short_circuit_synthesis(
    *, state: GraphState, debug: Any, context: SynthesisContext, stage_started: float,
) -> GraphState | None:
    packet = []
    if context.guided_followup:
        result = finalize_answer(text_document(context.guided_followup), [])
        mode = "guided_followup"
    elif not context.retrieval_required and is_action_only_request(context.user_input):
        if needs_slack(context.user_input) and not context.slack_target_available:
            result = finalize_answer(text_document("Slack으로 보낼 대상의 channel_id, user_id 또는 email을 알려주세요."), [])
            mode = "action_only"
        else:
            previous = get_runtime_state(state).previous_response
            if previous is None or not previous.content.blocks:
                return None
            packet = [citation.evidence for citation in previous.citations]
            result = finalize_answer(previous.content, packet, retrieval_required=previous.retrieval_required, issues=previous.issues)
            mode = "action_only"
    elif (
        context.retrieval_required and len(context.hits) == 1
        and len(context.planner_output.tasks) == 1
        and context.hits[0].evidence.route == "upload"
        and is_explicit_source_extraction(context.user_input)
        and any(d.requirement_id == context.hits[0].requirement_id and d.answerability == "covered"
                for d in debug.retrieval_diagnostics)
    ):
        packet = [context.hits[0].evidence]
        result = build_grounded_response(packet, message="요청한 자료의 원문 발췌입니다.")
        mode = "deterministic_grounded_direct"
    else:
        return None
    return build_synthesis_updates(
        debug=debug, result=result, evidence_packet=packet, attempt=context.attempt,
        latency_trace=[make_stage_latency_event(
            stage="synthesis", attempt=context.attempt,
            latency_ms=elapsed_ms(stage_started, time.perf_counter()), status=mode,
        )],
        retrieval_errors=context.parse_errors,
        planner_errors=context.planner_parse_errors,
        evidence_requirement_map={item.id: [hit.requirement_id for hit in context.hits if hit.evidence.id == item.id and hit.requirement_id]
                                  for item in packet},
    )
