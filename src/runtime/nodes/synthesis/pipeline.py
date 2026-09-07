from __future__ import annotations

import time
from typing import Any

from src.core.answer_schema import AnswerResponse, finalize_answer, iter_content_units
from src.core.contracts.debug import build_llm_call_metadata
from src.core.latency import elapsed_ms, make_stage_latency_event, make_synthesis_attempt_latency_event
from src.runtime.nodes.synthesis.fallbacks import build_synthesis_fallback
from src.runtime.nodes.synthesis.models import PreparedSynthesisInputs, SynthesisPipelineResult
from src.runtime.nodes.synthesis.schema_adapter import coerce_answer_document, coerce_structured_synthesis_result

_FALLBACK_NOTICE = "답변을 완성하지 못했습니다. 확인 가능한 원문 근거를 아래에 표시합니다."


def _is_timeout_error(exc: Exception) -> bool:
    return "timeout" in str(exc).lower() or "timed out" in str(exc).lower()


def _invoke_structured_attempt(
    *, structured_synthesizer: Any, prepared: PreparedSynthesisInputs, llm_calls: list[Any], path: str,
) -> AnswerResponse:
    structured = structured_synthesizer.invoke(prepared.model_messages)
    parsed, raw, error = coerce_structured_synthesis_result(structured)
    if raw is not None:
        llm_calls.append(build_llm_call_metadata(
            stage="synthesis", attempt=prepared.attempt, path=path, message=raw,
        ))
    if error is not None:
        raise error
    document = coerce_answer_document(parsed)
    if not document.blocks:
        raise ValueError("structured output was empty")
    if prepared.reference_aliases:
        document = document.model_copy(deep=True)
        for _path, unit in iter_content_units(document):
            unit.refs = [prepared.reference_aliases.get(reference, reference) for reference in unit.refs]
    return finalize_answer(document, prepared.evidence_packet, retrieval_required=prepared.retrieval_required)


def run_synthesis_pipeline(
    *, structured_synthesizer: Any, structured_synthesizer_compact: Any | None,
    prepared: PreparedSynthesisInputs, compact_prepared: PreparedSynthesisInputs | None,
    stage_started: float,
) -> SynthesisPipelineResult:
    errors: list[str] = []
    llm_calls: list[Any] = []
    started = time.perf_counter()
    structured_ms: int | None = None
    fallback_ms: int | None = None
    mode = "structured_only"
    used = prepared
    try:
        result = _invoke_structured_attempt(
            structured_synthesizer=structured_synthesizer, prepared=prepared,
            llm_calls=llm_calls, path="structured",
        )
        structured_ms = elapsed_ms(started, time.perf_counter())
    except Exception as exc:
        structured_ms = elapsed_ms(started, time.perf_counter())
        errors.append(f"synthesize: structured output failed ({exc})")
        fallback_started = time.perf_counter()
        if _is_timeout_error(exc) and structured_synthesizer_compact is not None and compact_prepared is not None:
            used = compact_prepared
            try:
                result = _invoke_structured_attempt(
                    structured_synthesizer=structured_synthesizer_compact, prepared=used,
                    llm_calls=llm_calls, path="structured_compact_fallback",
                )
                mode = "compact_structured_fallback"
            except Exception as compact_exc:
                errors.append(f"synthesize: compact structured output failed ({compact_exc})")
                result = build_synthesis_fallback(
                    evidence_packet=used.evidence_packet, retrieval_required=used.retrieval_required,
                    message=_FALLBACK_NOTICE,
                )
                mode = "timeout_grounded_fallback"
        else:
            result = build_synthesis_fallback(
                evidence_packet=used.evidence_packet, retrieval_required=used.retrieval_required,
                message=_FALLBACK_NOTICE,
            )
            mode = "timeout_grounded_fallback" if _is_timeout_error(exc) else "deterministic_grounded_fallback"
        fallback_ms = elapsed_ms(fallback_started, time.perf_counter())
    total = elapsed_ms(stage_started, time.perf_counter())
    return SynthesisPipelineResult(
        result=result, evidence_packet=used.evidence_packet,
        evidence_requirement_map=used.evidence_requirement_map,
        latency_trace=[
            make_synthesis_attempt_latency_event(
                attempt=prepared.attempt, mode=mode, structured_ms=structured_ms,
                fallback_ms=fallback_ms, total_ms=total,
            ),
            make_stage_latency_event(stage="synthesis", attempt=prepared.attempt, latency_ms=total, status=mode),
        ],
        retrieval_errors=prepared.parse_errors, planner_errors=prepared.planner_parse_errors,
        synthesis_errors=errors, llm_calls=llm_calls,
    )
