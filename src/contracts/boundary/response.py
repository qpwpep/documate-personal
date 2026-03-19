from __future__ import annotations

from typing import Any

from ...answer_schema import AgentResponsePayloadModel, SynthesisOutput, build_empty_response_payload
from ..graph_state import ResponseState


def parse_response_state(value: Any) -> ResponseState:
    if isinstance(value, ResponseState):
        return value
    if not isinstance(value, dict):
        return ResponseState()

    raw_payload = value.get("payload")
    if isinstance(raw_payload, AgentResponsePayloadModel):
        payload = raw_payload
    else:
        try:
            payload = AgentResponsePayloadModel.model_validate(raw_payload)
        except Exception:
            payload = build_empty_response_payload(answer="")

    raw_synthesis_output = value.get("synthesis_output")
    if isinstance(raw_synthesis_output, SynthesisOutput):
        synthesis_output = raw_synthesis_output
    else:
        try:
            synthesis_output = SynthesisOutput.model_validate(raw_synthesis_output)
        except Exception:
            synthesis_output = SynthesisOutput()

    synthesis_attempt = value.get("synthesis_attempt", 0)
    try:
        synthesis_attempt_int = int(synthesis_attempt or 0)
    except (TypeError, ValueError):
        synthesis_attempt_int = 0

    return ResponseState(
        final_answer=str(value.get("final_answer", "") or ""),
        payload=payload,
        synthesis_output=synthesis_output,
        synthesis_attempt=max(0, synthesis_attempt_int),
    )


def get_response_state(state: dict[str, Any]) -> ResponseState:
    return parse_response_state(state.get("response"))
