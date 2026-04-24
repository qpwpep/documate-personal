from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from src.core.answer_schema import AgentResponsePayloadModel, SynthesisOutput
from src.core.contracts import GraphState, ResponseState
from src.core.contracts.debug import LLMCallMetadata


def _error_codes_from_synthesis_errors(errors: list[str] | None) -> list[str]:
    codes: list[str] = []
    for error in errors or []:
        lowered = str(error or "").lower()
        if "structured output was empty" in lowered and "LLM_STRUCTURED_EMPTY" not in codes:
            codes.append("LLM_STRUCTURED_EMPTY")
        if ("timeout" in lowered or "timed out" in lowered) and "SYNTHESIS_TIMEOUT" not in codes:
            codes.append("SYNTHESIS_TIMEOUT")
    return codes


def build_response_state(
    *,
    payload: AgentResponsePayloadModel,
    synthesis_output: SynthesisOutput,
    final_answer: str,
    attempt: int,
) -> ResponseState:
    return ResponseState(
        final_answer=final_answer,
        payload=payload,
        synthesis_output=synthesis_output,
        synthesis_attempt=attempt,
    )


def build_synthesis_updates(
    *,
    debug: Any,
    payload: AgentResponsePayloadModel,
    synthesis_output: SynthesisOutput,
    final_answer: str,
    attempt: int,
    latency_trace: list[dict[str, Any]],
    retrieval_errors: list[str] | None = None,
    planner_errors: list[str] | None = None,
    synthesis_errors: list[str] | None = None,
    llm_calls: list[LLMCallMetadata] | None = None,
) -> GraphState:
    return {
        "messages": [AIMessage(content=final_answer)],
        "response": build_response_state(
            payload=payload,
            synthesis_output=synthesis_output,
            final_answer=final_answer,
            attempt=attempt,
        ),
        "debug": debug.model_copy(
            update={
                "retrieval_errors": [*debug.retrieval_errors, *(retrieval_errors or [])],
                "planner_errors": [*debug.planner_errors, *(planner_errors or [])],
                "synthesis_errors": [*debug.synthesis_errors, *(synthesis_errors or [])],
                "error_codes": [
                    *debug.error_codes,
                    *[
                        code
                        for code in _error_codes_from_synthesis_errors(synthesis_errors)
                        if code not in debug.error_codes
                    ],
                ],
                "llm_calls": [*debug.llm_calls, *(llm_calls or [])],
                "latency_trace": [*debug.latency_trace, *latency_trace],
            }
        ),
    }
