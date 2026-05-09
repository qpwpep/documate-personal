from __future__ import annotations

import time
from typing import Any

from src.core.answer_schema import SynthesisOutput, build_empty_response_payload, normalize_answer_sections
from src.core.contracts.debug import build_llm_call_metadata
from src.core.latency import elapsed_ms, make_stage_latency_event, make_synthesis_attempt_latency_event
from src.infra.tail_latency import invoke_with_optional_hedge
from src.runtime.nodes.synthesis.fallback_renderers import RenderedSynthesisPayload, build_local_fallback_payload, enforce_synthesis_output_budget, render_synthesis_payload
from src.runtime.nodes.synthesis.models import PreparedSynthesisInputs, SynthesisPipelineResult
from src.runtime.nodes.synthesis.schema_adapter import coerce_structured_synthesis_result, coerce_synthesis_output


_DEFAULT_GENERIC_FALLBACK_ANSWER = (
    "응답 생성 중 오류가 발생했습니다. "
    "질문 범위를 조금 좁혀 다시 시도해 주세요."
)


def _is_timeout_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "timeout" in lowered or "timed out" in lowered


def _build_rendered_payload_from_payload(payload: Any) -> RenderedSynthesisPayload:
    return RenderedSynthesisPayload(
        payload=payload,
        final_answer=payload.answer,
        synthesis_output=SynthesisOutput(
            answer=payload.answer,
            claims=payload.claims,
            confidence=payload.confidence,
            sections=payload.sections,
        ),
    )


def _is_structured_synthesis_success(result: Any) -> bool:
    raw_response_obj, _raw_message, structured_error = coerce_structured_synthesis_result(result)
    if structured_error is not None:
        return False
    output = coerce_synthesis_output(raw_response_obj)
    has_renderable_section = bool(normalize_answer_sections(output.sections))
    return bool(
        str(output.answer or "").strip()
        or output.claims
        or has_renderable_section
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


def _invoke_structured_attempt(
    *,
    structured_synthesizer: Any,
    prepared: PreparedSynthesisInputs,
    llm_calls: list[Any],
    path: str,
    hedge_delay_seconds: float = 0.0,
    hedge_max_attempts: int = 2,
    hedge_overall_timeout_seconds: float | None = None,
) -> tuple[RenderedSynthesisPayload, int]:
    attempt_started = time.perf_counter()
    hedge_result = invoke_with_optional_hedge(
        lambda: structured_synthesizer.invoke(prepared.model_messages),
        hedge_delay_seconds=hedge_delay_seconds,
        max_attempts=hedge_max_attempts,
        is_success=_is_structured_synthesis_success,
        overall_timeout_seconds=hedge_overall_timeout_seconds,
    )
    structured_result = hedge_result.value
    attempt_ms = elapsed_ms(attempt_started, time.perf_counter())
    raw_response_obj, raw_message, structured_error = coerce_structured_synthesis_result(
        structured_result
    )
    if raw_message is not None:
        call_metadata = build_llm_call_metadata(
            stage="synthesis",
            attempt=prepared.attempt,
            path=path,  # type: ignore[arg-type]
            message=raw_message,
        )
        if hedge_result.hedge_dropped:
            call_metadata = call_metadata.model_copy(
                update={
                        "response_metadata": {
                            **call_metadata.response_metadata,
                            "hedge_dropped": True,
                            "hedge_attempts_started": hedge_result.hedges_started,
                            "hedge_attempts_dropped": hedge_result.hedges_dropped,
                        }
                    }
                )
        llm_calls.append(call_metadata)
        if hedge_result.hedge_started:
            llm_calls.append(
                call_metadata.model_copy(
                    update={
                        "path": "structured_hedge",
                        "response_metadata": {
                            **call_metadata.response_metadata,
                            "hedge_winner": hedge_result.winner,
                            "hedge_dropped": hedge_result.hedge_dropped,
                            "hedge_attempts_started": hedge_result.hedges_started,
                            "hedge_attempts_dropped": hedge_result.hedges_dropped,
                            "hedge_duplicate_estimate": True,
                        },
                    }
                )
            )
    if structured_error is not None:
        raise structured_error
    return (
        render_synthesis_payload(
            coerce_synthesis_output(raw_response_obj),
            prepared.primary_evidence_items,
        ),
        attempt_ms,
    )


def run_synthesis_pipeline(
    *,
    structured_synthesizer: Any,
    structured_synthesizer_compact: Any | None,
    prepared: PreparedSynthesisInputs,
    compact_prepared: PreparedSynthesisInputs | None,
    stage_started: float,
    hedge_delay_seconds: float = 0.0,
    hedge_max_attempts: int = 2,
    hedge_overall_timeout_seconds: float | None = None,
) -> SynthesisPipelineResult:
    synthesis_errors: list[str] = []
    llm_calls = []
    structured_ms: int | None = None
    fallback_ms: int | None = None
    synthesis_mode = "structured_only"

    structured_started = time.perf_counter()
    try:
        rendered, structured_ms = _invoke_structured_attempt(
            structured_synthesizer=structured_synthesizer,
            prepared=prepared,
            llm_calls=llm_calls,
            path="structured",
            hedge_delay_seconds=hedge_delay_seconds,
            hedge_max_attempts=hedge_max_attempts,
            hedge_overall_timeout_seconds=hedge_overall_timeout_seconds,
        )
        rendered, used_empty_fallback = _ensure_non_empty_rendered_payload(
            rendered=rendered,
            prepared=prepared,
            generic_answer=_DEFAULT_GENERIC_FALLBACK_ANSWER,
        )
        rendered = enforce_synthesis_output_budget(
            rendered=rendered,
            evidence_items=prepared.primary_evidence_items,
            budget_profile=prepared.budget_profile,
        )
        if used_empty_fallback:
            synthesis_errors.append("synthesize: structured output was empty")
            fallback_ms = 0
            synthesis_mode = "structured_empty_fallback"
    except Exception as exc:
        if structured_ms is None:
            structured_ms = elapsed_ms(structured_started, time.perf_counter())
        synthesis_errors.append(
            (
                f"synthesize: structured output timed out ({exc})"
                if _is_timeout_error(exc)
                else f"synthesize: structured output failed ({exc})"
            )
        )
        is_timeout = _is_timeout_error(exc)
        if is_timeout and structured_synthesizer_compact is not None and compact_prepared is not None:
            compact_started = time.perf_counter()
            try:
                rendered, _compact_ms = _invoke_structured_attempt(
                    structured_synthesizer=structured_synthesizer_compact,
                    prepared=compact_prepared,
                    llm_calls=llm_calls,
                    path="structured_compact_fallback",
                )
                rendered, used_empty_fallback = _ensure_non_empty_rendered_payload(
                    rendered=rendered,
                    prepared=prepared,
                    generic_answer=_DEFAULT_GENERIC_FALLBACK_ANSWER,
                )
                rendered = enforce_synthesis_output_budget(
                    rendered=rendered,
                    evidence_items=prepared.primary_evidence_items,
                    budget_profile=prepared.budget_profile,
                )
                fallback_ms = elapsed_ms(compact_started, time.perf_counter())
                if used_empty_fallback:
                    synthesis_errors.append("synthesize: compact structured output was empty")
                    synthesis_mode = "timeout_grounded_fallback"
                else:
                    synthesis_mode = "compact_structured_fallback"
            except Exception as compact_exc:
                synthesis_errors.append(
                    (
                        f"synthesize: compact structured output timed out ({compact_exc})"
                        if _is_timeout_error(compact_exc)
                        else f"synthesize: compact structured output failed ({compact_exc})"
                    )
                )
                fallback_payload = build_local_fallback_payload(
                    evidence_items=prepared.grounded_fallback_evidence_items,
                    retrieval_required=prepared.retrieval_required,
                    generic_answer=_DEFAULT_GENERIC_FALLBACK_ANSWER,
                )
                rendered = _build_rendered_payload_from_payload(fallback_payload)
                fallback_ms = elapsed_ms(compact_started, time.perf_counter())
                synthesis_mode = "timeout_grounded_fallback"
        else:
            fallback_started = time.perf_counter()
            fallback_payload = build_local_fallback_payload(
                evidence_items=prepared.grounded_fallback_evidence_items,
                retrieval_required=prepared.retrieval_required,
                generic_answer=_DEFAULT_GENERIC_FALLBACK_ANSWER,
            )
            fallback_ms = elapsed_ms(fallback_started, time.perf_counter())
            rendered = _build_rendered_payload_from_payload(fallback_payload)
            synthesis_mode = (
                "timeout_grounded_fallback" if is_timeout else "deterministic_grounded_fallback"
            )

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
