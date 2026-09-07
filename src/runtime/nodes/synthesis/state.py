from __future__ import annotations

from typing import Any

from src.core.answer_schema import AnswerResponse
from src.core.contracts import GraphState, ResponseState
from src.core.contracts.debug import LLMCallMetadata
from src.core.evidence import EvidenceRef


def _error_codes_from_synthesis_errors(errors: list[str] | None) -> list[str]:
    codes: list[str] = []
    for error in errors or []:
        lowered = error.lower()
        if "structured output was empty" in lowered and "LLM_STRUCTURED_EMPTY" not in codes:
            codes.append("LLM_STRUCTURED_EMPTY")
        if ("timeout" in lowered or "timed out" in lowered) and "SYNTHESIS_TIMEOUT" not in codes:
            codes.append("SYNTHESIS_TIMEOUT")
    return codes


def build_synthesis_updates(
    *, debug: Any, result: AnswerResponse, evidence_packet: list[EvidenceRef], attempt: int,
    latency_trace: list[dict[str, Any]], retrieval_errors: list[str] | None = None,
    planner_errors: list[str] | None = None, synthesis_errors: list[str] | None = None,
    llm_calls: list[LLMCallMetadata] | None = None,
    evidence_requirement_map: dict[str, list[str]] | None = None,
) -> GraphState:
    return {
        "response": ResponseState(result=result, evidence_packet=evidence_packet, synthesis_attempt=attempt,
                                  evidence_requirement_map=evidence_requirement_map or {}),
        "debug": debug.model_copy(update={
            "retrieval_errors": [*debug.retrieval_errors, *(retrieval_errors or [])],
            "planner_errors": [*debug.planner_errors, *(planner_errors or [])],
            "synthesis_errors": [*debug.synthesis_errors, *(synthesis_errors or [])],
            "error_codes": list(dict.fromkeys([*debug.error_codes, *_error_codes_from_synthesis_errors(synthesis_errors)])),
            "llm_calls": [*debug.llm_calls, *(llm_calls or [])],
            "latency_trace": [*debug.latency_trace, *latency_trace],
        }),
    }
