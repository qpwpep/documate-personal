from __future__ import annotations

import time
from typing import Any

from ...answer_schema import SynthesisOutput, build_empty_response_payload
from ...contracts.debug import build_llm_call_metadata
from ...latency import elapsed_ms, make_stage_latency_event, make_synthesis_attempt_latency_event
from .models import PreparedSynthesisInputs, SynthesisPipelineResult
from .payload_builder import (
    RenderedSynthesisPayload,
    build_local_fallback_payload,
    coerce_structured_synthesis_result,
    coerce_synthesis_output,
    render_synthesis_payload,
)


_DEFAULT_GENERIC_FALLBACK_ANSWER = (
    "응답 생성 중 오류가 발생했습니다. "
    "질문 범위를 조금 좁혀 다시 시도해 주세요."
)


def _build_rendered_payload_from_payload(payload: Any) -> RenderedSynthesisPayload:
    return RenderedSynthesisPayload(
        payload=payload,
        final_answer=payload.answer,
        synthesis_output=SynthesisOutput(
            answer=payload.answer,
            claims=payload.claims,
            confidence=payload.confidence,
        ),
    )


def _ensure_non_empty_rendered_payload(
    *,
    rendered: RenderedSynthesisPayload,
    prepared: PreparedSynthesisInputs,
    generic_answer: str,
) -> tuple[RenderedSynthesisPayload, bool]:
    if rendered.payload.claims or str(rendered.final_answer or "").strip():
        return rendered, False

    fallback_payload = build_local_fallback_payload(
        evidence_items=prepared.grounded_fallback_evidence_items,
        retrieval_required=prepared.retrieval_required,
        generic_answer=generic_answer,
    )
    if not fallback_payload.claims and not str(fallback_payload.answer or "").strip():
        fallback_payload = build_empty_response_payload(answer=generic_answer)
    return _build_rendered_payload_from_payload(fallback_payload), True


def run_synthesis_pipeline(
    *,
    structured_synthesizer: Any,
    prepared: PreparedSynthesisInputs,
    stage_started: float,
) -> SynthesisPipelineResult:
    synthesis_errors: list[str] = []
    llm_calls = []
    structured_ms: int | None = None
    fallback_ms: int | None = None
    synthesis_mode = "structured_only"

    try:
        structured_started = time.perf_counter()
        structured_result = structured_synthesizer.invoke(prepared.model_messages)
        structured_ms = elapsed_ms(structured_started, time.perf_counter())
        raw_response_obj, raw_message, structured_error = coerce_structured_synthesis_result(
            structured_result
        )
        if raw_message is not None:
            llm_calls.append(
                build_llm_call_metadata(
                    stage="synthesis",
                    attempt=prepared.attempt,
                    path="structured",
                    message=raw_message,
                )
            )
        if structured_error is not None:
            raise structured_error
        rendered = render_synthesis_payload(
            coerce_synthesis_output(raw_response_obj),
            prepared.primary_evidence_items,
        )
        rendered, used_empty_fallback = _ensure_non_empty_rendered_payload(
            rendered=rendered,
            prepared=prepared,
            generic_answer=_DEFAULT_GENERIC_FALLBACK_ANSWER,
        )
        if used_empty_fallback:
            synthesis_errors.append("synthesize: structured output was empty")
            fallback_ms = 0
            synthesis_mode = "structured_empty_fallback"
    except Exception as exc:
        structured_ms = elapsed_ms(structured_started, time.perf_counter())
        synthesis_errors.append(
            (
                f"synthesize: structured output timed out ({exc})"
                if "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
                else f"synthesize: structured output failed ({exc})"
            )
        )
        fallback_started = time.perf_counter()
        fallback_payload = build_local_fallback_payload(
            evidence_items=prepared.grounded_fallback_evidence_items,
            retrieval_required=prepared.retrieval_required,
            generic_answer=_DEFAULT_GENERIC_FALLBACK_ANSWER,
        )
        fallback_ms = elapsed_ms(fallback_started, time.perf_counter())
        rendered = _build_rendered_payload_from_payload(fallback_payload)
        synthesis_mode = "deterministic_grounded_fallback"

    total_ms = elapsed_ms(stage_started, time.perf_counter())
    return SynthesisPipelineResult(
        payload=rendered.payload,
        synthesis_output=rendered.synthesis_output,
        final_answer=rendered.final_answer,
        latency_trace=[
            make_synthesis_attempt_latency_event(
                attempt=prepared.attempt,
                mode=synthesis_mode,  # type: ignore[arg-type]
                structured_ms=structured_ms,
                fallback_ms=fallback_ms,
                total_ms=total_ms,
            ),
            make_stage_latency_event(
                stage="synthesis",
                attempt=prepared.attempt,
                latency_ms=total_ms,
                status=synthesis_mode,
            ),
        ],
        retrieval_errors=prepared.parse_errors,
        planner_errors=prepared.planner_parse_errors,
        synthesis_errors=synthesis_errors,
        llm_calls=llm_calls,
    )
