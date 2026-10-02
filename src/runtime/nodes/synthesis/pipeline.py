from __future__ import annotations

import time
from typing import Any

from src.core.answer_schema import AnswerResponse, finalize_answer, iter_content_units
from src.core.latency import elapsed_ms, make_stage_latency_event, make_synthesis_attempt_latency_event
from src.core.llm_errors import LLMCallError, LLMDiagnostic
from src.infra.llm_boundary import CallBudget, is_timeout_failure, request_call_budget, run_structured_call
from src.runtime.nodes.synthesis.fallbacks import build_synthesis_fallback
from src.runtime.nodes.synthesis.models import PreparedSynthesisInputs, SynthesisPipelineResult
from src.runtime.nodes.synthesis.schema_adapter import coerce_answer_document

_RECOVERABLE_FAILURES = {"provider_unavailable", "provider_rate_limited", "model_output_invalid", "model_output_incomplete"}


def _invoke_structured_attempt(
    *, structured_synthesizer: Any, prepared: PreparedSynthesisInputs, path: str,
    budget: CallBudget | None = None, retry_timeouts: bool = True,
) -> AnswerResponse:
    def validate(parsed: Any) -> AnswerResponse:
        document = coerce_answer_document(parsed)
        if not document.blocks:
            raise ValueError("structured output was empty")
        if prepared.reference_aliases:
            document = document.model_copy(deep=True)
            for _path, unit in iter_content_units(document):
                unit.refs = [prepared.reference_aliases.get(reference, reference) for reference in unit.refs]
        return finalize_answer(document, prepared.evidence_packet, retrieval_required=prepared.retrieval_required)

    return run_structured_call(
        structured_synthesizer, prepared.model_messages, stage="synthesis", validate=validate,
        budget=budget, path=path, attempt=prepared.attempt, retry_timeouts=retry_timeouts,
    )


def _grounded_failure(error: LLMCallError, prepared: PreparedSynthesisInputs) -> AnswerResponse:
    """Recovery may expose checked source excerpts, never invent a user input problem."""
    if error.problem.code not in _RECOVERABLE_FAILURES or not prepared.evidence_packet:
        raise error
    result = build_synthesis_fallback(
        evidence_packet=prepared.evidence_packet, retrieval_required=prepared.retrieval_required,
        message=f"{error.problem.message} 확인 가능한 원문 근거만 아래에 표시합니다.",
        request_contract=prepared.request_contract,
    )
    if not result.citations:
        # The fallback could not satisfy the existing contract. Preserve the original cause.
        raise error
    return result


def run_synthesis_pipeline(
    *, structured_synthesizer: Any, structured_synthesizer_compact: Any | None,
    prepared: PreparedSynthesisInputs, compact_prepared: PreparedSynthesisInputs | None,
    stage_started: float,
) -> SynthesisPipelineResult:
    errors: list[str] = []
    diagnostics: list[LLMDiagnostic] = []
    problem = None
    budget = request_call_budget("synthesis")
    started = time.perf_counter()
    structured_ms: int | None = None
    fallback_ms: int | None = None
    mode = "structured_only"
    used = prepared
    has_compact = structured_synthesizer_compact is not None and compact_prepared is not None
    try:
        result = _invoke_structured_attempt(
            structured_synthesizer=structured_synthesizer, prepared=prepared,
            path="structured", budget=budget,
            retry_timeouts=not has_compact,
        )
        structured_ms = elapsed_ms(started, time.perf_counter())
    except LLMCallError as exc:
        structured_ms = elapsed_ms(started, time.perf_counter())
        errors.append(f"synthesize: {exc.problem.code}")
        diagnostics.append(exc.diagnostic)
        fallback_started = time.perf_counter()
        if is_timeout_failure(exc) and has_compact and budget.can_attempt:
            used = compact_prepared
            try:
                result = _invoke_structured_attempt(
                    structured_synthesizer=structured_synthesizer_compact, prepared=used,
                    path="structured_compact_fallback", budget=budget,
                )
                mode = "compact_structured_fallback"
            except LLMCallError as compact_exc:
                errors.append(f"synthesize: {compact_exc.problem.code}")
                diagnostics.append(compact_exc.diagnostic)
                result = _grounded_failure(compact_exc, used)
                problem = compact_exc.problem
                mode = "timeout_grounded_fallback"
        else:
            result = _grounded_failure(exc, used)
            problem = exc.problem
            mode = "timeout_grounded_fallback" if exc.problem.code == "provider_unavailable" else "deterministic_grounded_fallback"
        fallback_ms = elapsed_ms(fallback_started, time.perf_counter())
    total = elapsed_ms(stage_started, time.perf_counter())
    return SynthesisPipelineResult(
        result=result, evidence_packet=used.evidence_packet,
        evidence_requirement_map=used.evidence_requirement_map,
        normal_evidence_missing_requirement_ids=prepared.missing_requirement_ids,
        latency_trace=[
            make_synthesis_attempt_latency_event(
                attempt=prepared.attempt, mode=mode, structured_ms=structured_ms,
                fallback_ms=fallback_ms, total_ms=total,
            ),
            make_stage_latency_event(stage="synthesis", attempt=prepared.attempt, latency_ms=total, status=mode),
        ],
        retrieval_errors=prepared.parse_errors, planner_errors=prepared.planner_parse_errors,
        synthesis_errors=errors,
        problem=problem, llm_diagnostics=diagnostics,
        kind="failure" if mode in {"timeout_grounded_fallback", "deterministic_grounded_fallback"} else "draft",
    )
