from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage

from ...answer_schema import SynthesisOutput
from ...contracts.debug import build_llm_call_metadata
from ...latency import elapsed_ms, make_stage_latency_event, make_synthesis_attempt_latency_event
from ..session import extract_text_content
from .models import PreparedSynthesisInputs, SynthesisPipelineResult
from .payload_builder import (
    RenderedSynthesisPayload,
    build_local_fallback_payload,
    build_plain_summary_attach_payload,
    coerce_structured_synthesis_result,
    coerce_synthesis_output,
    render_synthesis_payload,
)
from .prompt_builder import build_plain_summary_attach_messages


_DEFAULT_GENERIC_FALLBACK_ANSWER = (
    "\uc751\ub2f5 \uc0dd\uc131 \uc911 \uc624\ub958\uac00 \ubc1c\uc0dd\ud588\uc2b5\ub2c8\ub2e4. "
    "\uc9c8\ubb38 \ubc94\uc704\ub97c \uc870\uae08 \uc881\ud600 \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \uc8fc\uc138\uc694."
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


def run_synthesis_pipeline(
    *,
    structured_synthesizer: Any,
    fallback_llm: Any,
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
    except Exception as exc:
        structured_ms = elapsed_ms(structured_started, time.perf_counter())
        synthesis_errors.append(
            (
                f"synthesize: structured output timed out ({exc})"
                if "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
                else f"synthesize: structured output failed ({exc})"
            )
        )
        try:
            fallback_started = time.perf_counter()
            fallback_result = fallback_llm.invoke(
                build_plain_summary_attach_messages(
                    user_input=prepared.user_input,
                    deduped_evidence=prepared.deduped_evidence,
                )
            )
            fallback_ms = elapsed_ms(fallback_started, time.perf_counter())
            if isinstance(fallback_result, AIMessage):
                llm_calls.append(
                    build_llm_call_metadata(
                        stage="synthesis",
                        attempt=prepared.attempt,
                        path="plain_summary_attach_fallback",
                        message=fallback_result,
                    )
                )
            fallback_payload = build_plain_summary_attach_payload(
                content=extract_text_content(getattr(fallback_result, "content", fallback_result)),
                evidence_items=prepared.primary_evidence_items,
            )
            if fallback_payload is None:
                raise RuntimeError("plain summary attach payload could not be built")
            rendered = _build_rendered_payload_from_payload(fallback_payload)
            synthesis_mode = "plain_summary_attach_fallback"
        except Exception as fallback_exc:
            if fallback_ms is None:
                fallback_ms = elapsed_ms(fallback_started, time.perf_counter())
            synthesis_errors.append(f"synthesize: plain summary attach failed ({fallback_exc})")
            fallback_payload = build_local_fallback_payload(
                evidence_items=prepared.grounded_fallback_evidence_items,
                retrieval_required=prepared.retrieval_required,
                generic_answer=_DEFAULT_GENERIC_FALLBACK_ANSWER,
            )
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
