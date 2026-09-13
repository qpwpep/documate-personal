from __future__ import annotations

from src.core.answer_schema import ActionReceipt, iter_content_units
from ..judge_llm import LLMJudge
from ..evidence_scope import assess_evidence_scope
from ..metric_rules import compute_rule_scores
from ..pricing import compute_cost_usd
from ..config_models import BenchmarkCase, BenchmarkConfig
from ..result_models import CaseResult, JudgeSubscores, ScenarioTurnResult
from ..weighting import (
    compute_composite_quality_score,
    compute_rule_weighted_score,
    resolve_base_weights_for_case,
    resolve_effective_weights,
)
from .response_parser import ParsedResponseData


_DEFAULT_JUDGE_MIN_SCORES: dict[str, float] = {
    "docs_only": 0.70,
    "hybrid": 0.70,
}
_PRODUCT_PASS_FLOOR = 0.75


def _resolve_judge_min_score(case: BenchmarkCase, config: BenchmarkConfig) -> float | None:
    if case.judge_min_score is not None:
        return float(case.judge_min_score)
    configured_threshold = config.judge_min_score.for_category(case.category)
    if configured_threshold is not None:
        return float(configured_threshold)
    return _DEFAULT_JUDGE_MIN_SCORES.get(case.category)


def _build_gate_failures(
    *,
    runtime_errors: list[str],
    response_errors: list[str],
    debug_errors: list[str],
    missing_required_debug_fields: list[str],
    product_pass: bool | None,
    judge_pass: bool | None,
    judge_errors: list[str],
    judge_audit_failures: list[str],
) -> list[str]:
    failures: list[str] = []
    if runtime_errors:
        failures.append("runtime_error")
    if response_errors:
        failures.append("response_contract_error")
    if debug_errors:
        failures.append("debug_error")
    if missing_required_debug_fields:
        failures.append("missing_debug_fields")
    if product_pass is False:
        failures.append("product_quality_below_floor")
    if judge_pass is False:
        failures.append("judge_min_score_audit_failed")
    if judge_audit_failures and "judge_min_score_audit_failed" not in failures:
        failures.append("judge_min_score_audit_failed")
    if any(str(error).startswith("invalid_eval:") for error in judge_errors):
        failures.append("invalid_eval")
    if any("judge payload is incomplete" in str(error) for error in judge_errors):
        failures.append("judge_input_incomplete")
    return failures


def _resolve_slack_delivery_status(
    *,
    slack_delivery_required: bool,
    actions: list[ActionReceipt],
) -> tuple[str, str | None]:
    if not slack_delivery_required:
        return "not_applicable", None
    receipt = next((action for action in reversed(actions) if action.kind == "slack_notify"), None)
    if receipt is None:
        return "unknown", "missing_action_receipt"
    if receipt.status == "success":
        return "success", None
    if receipt.status == "skipped":
        return "skipped", receipt.message or receipt.error
    return "failed", receipt.error or receipt.message


def _extract_output_tokens(parsed_response: ParsedResponseData) -> int:
    if parsed_response.token_usage is not None and parsed_response.token_usage.completion_tokens > 0:
        return int(parsed_response.token_usage.completion_tokens)
    total = 0
    for call in parsed_response.llm_calls:
        if str(call.stage) != "synthesis":
            continue
        usage = call.usage_metadata or {}
        response_usage = call.response_metadata.get("token_usage")
        if not isinstance(response_usage, dict):
            response_usage = {}
        raw_output = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or response_usage.get("completion_tokens")
            or response_usage.get("output_tokens")
            or 0
        )
        try:
            total += max(0, int(raw_output or 0))
        except (TypeError, ValueError):
            continue
    return total


def build_case_result(
    *,
    run_id: str,
    endpoint_url: str,
    case: BenchmarkCase,
    judge: LLMJudge,
    config: BenchmarkConfig,
    session_id: str,
    created_at: str,
    request_payload: dict,
    latency_ms_e2e: int | None,
    parsed_response: ParsedResponseData,
    slack_delivery_required: bool = False,
    prior_turns: list[ScenarioTurnResult] | None = None,
) -> CaseResult:
    effective_weights, weights_error = resolve_effective_weights(
        case=case,
        base_weights=resolve_base_weights_for_case(
            case=case,
            base_weights=config.weights,
        ),
        case_override=case.weight_override,
    )
    runtime_errors = list(parsed_response.runtime_errors)
    response_errors = list(parsed_response.response_errors)
    if weights_error:
        runtime_errors.append(f"weight_override error: {weights_error}")

    response = parsed_response.response
    output_tokens = _extract_output_tokens(parsed_response)
    checks = response.checks if response is not None else []
    resolved_unit_count = sum(check.reference_status == "resolved" for check in checks)
    missing_reference_unit_count = sum(check.reference_status == "missing" for check in checks)
    factual_paths = {path for path, unit in iter_content_units(response.content) if unit.basis != "interaction"} if response is not None else set()
    unchecked_unit_count = sum(check.unit_id in factual_paths and check.support_status == "not_evaluated" for check in checks)
    exact_match_unit_count = sum(check.support_status == "exact_match" for check in checks)
    unsupported_unit_count = sum(check.support_status == "unsupported" for check in checks)
    block_count = len(response.content.blocks) if response is not None else 0
    retrieval_warnings = sorted(
        {
            str(warning).strip()
            for diagnostic in parsed_response.retrieval_diagnostics
            for warning in diagnostic.warnings
            if str(warning).strip()
        }
    )
    validator_feedback = parsed_response.validator_feedback
    if retrieval_warnings:
        warning_text = ", ".join(retrieval_warnings)
        if validator_feedback:
            validator_feedback = f"{validator_feedback} | retrieval_warnings={warning_text}"
        else:
            validator_feedback = f"retrieval_warnings={warning_text}"
    slack_delivery_status, slack_delivery_error = _resolve_slack_delivery_status(
        slack_delivery_required=slack_delivery_required,
        actions=parsed_response.actions,
    )

    judge_errors: list[str] = []
    judge_audit_failures: list[str] = []
    llm_judge_score: float | None = None
    llm_judge_reason: str | None = None
    judge_subscores: JudgeSubscores | None = None
    judge_input_complete: bool | None = None
    evidence_scope = parsed_response.evidence_assessment
    if evidence_scope is None and response is not None and (
        parsed_response.answer_provenance is not None or prior_turns is not None
    ):
        evidence_scope = assess_evidence_scope(
            response=response, provenance=parsed_response.answer_provenance,
            observed_hits=parsed_response.observed_hits, tool_calls=parsed_response.tool_calls,
            session_id=session_id, prior_turns=prior_turns or [],
        )
    if evidence_scope is not None:
        for error in evidence_scope.errors:
            message = f"evidence scope: {error}"
            if error not in response_errors and message not in response_errors:
                response_errors.append(message)
    conversation = [{"query": turn.query, "response": turn.response,
                     "observed_hits": (turn.debug or {}).get("observed_hits", [])} for turn in (prior_turns or [])]
    if parsed_response.response_text.strip() and config.judge_enabled:
        judge_payload = judge.build_case_payload(
            case=case,
            tool_calls=parsed_response.tool_calls,
            response=response,
            observed_hits=parsed_response.observed_hits,
            retrieval_diagnostics=parsed_response.retrieval_diagnostics,
            planner_diagnostics=parsed_response.planner_diagnostics,
            validator_reason=parsed_response.validator_reason,
            synthesis_mode=parsed_response.synthesis_mode,
            resolved_unit_count=resolved_unit_count,
            missing_reference_unit_count=missing_reference_unit_count,
            unchecked_unit_count=unchecked_unit_count,
            tool_call_count=parsed_response.tool_call_count,
            slack_delivery_required=slack_delivery_required,
            conversation=conversation,
            evidence_scope=evidence_scope.model_dump(mode="json") if evidence_scope is not None else None,
            answer_provenance=parsed_response.answer_provenance,
        )
        judge_input_complete = judge.is_payload_complete(judge_payload)
        llm_judge_score, llm_judge_reason, judge_error, judge_subscores = judge.score_case(
            case=case,
            tool_calls=parsed_response.tool_calls,
            response=response,
            observed_hits=parsed_response.observed_hits,
            retrieval_diagnostics=parsed_response.retrieval_diagnostics,
            planner_diagnostics=parsed_response.planner_diagnostics,
            validator_reason=parsed_response.validator_reason,
            synthesis_mode=parsed_response.synthesis_mode,
            resolved_unit_count=resolved_unit_count,
            missing_reference_unit_count=missing_reference_unit_count,
            unchecked_unit_count=unchecked_unit_count,
            tool_call_count=parsed_response.tool_call_count,
            slack_delivery_required=slack_delivery_required,
            conversation=conversation,
            evidence_scope=evidence_scope.model_dump(mode="json") if evidence_scope is not None else None,
            answer_provenance=parsed_response.answer_provenance,
        )
        if judge_error:
            judge_errors.append(judge_error)

    rule_scores = compute_rule_scores(
        case=case,
        response=response,
        called_tools=parsed_response.tool_calls,
        observed_hits=parsed_response.observed_hits,
        runtime_errors=runtime_errors,
        response_errors=response_errors,
        judge_errors=judge_errors,
        validator_reason=parsed_response.validator_reason,
        synthesis_mode=parsed_response.synthesis_mode,
        slack_delivery_required=slack_delivery_required,
        slack_delivery_status=slack_delivery_status,
        evidence_scope=evidence_scope,
    )
    rule_weighted = compute_rule_weighted_score(rule_scores, effective_weights)

    composite_quality_score = compute_composite_quality_score(
        rule_weighted_score=rule_weighted,
        llm_judge_score=llm_judge_score,
        weights=effective_weights,
    )
    judge_min_score = _resolve_judge_min_score(case, config)
    judge_gate_passed: bool | None = None
    if judge_min_score is not None and parsed_response.response_text.strip():
        judge_gate_passed = False if llm_judge_score is None else llm_judge_score >= judge_min_score
    if judge_min_score is not None and llm_judge_score is not None and parsed_response.response_text.strip():
        judge_gate_passed = llm_judge_score >= judge_min_score
        if judge_gate_passed is False:
            judge_audit_failures.append(
                "judge_min_score audit failed: "
                f"score={llm_judge_score:.3f} threshold={judge_min_score:.3f}"
            )
    product_pass = composite_quality_score >= _PRODUCT_PASS_FLOOR
    if any(str(error).startswith("invalid_eval:") for error in judge_errors):
        judge_pass = False
    else:
        judge_pass = judge_gate_passed if judge_min_score is not None else (
            True if (config.judge_enabled and not judge_errors and llm_judge_score is not None) else None
        )
    release_pass = bool(product_pass and not runtime_errors and not response_errors)
    gate_failures = _build_gate_failures(
        runtime_errors=runtime_errors,
        response_errors=response_errors,
        debug_errors=parsed_response.debug_errors,
        missing_required_debug_fields=parsed_response.missing_required_debug_fields,
        product_pass=product_pass,
        judge_pass=judge_pass,
        judge_errors=judge_errors,
        judge_audit_failures=judge_audit_failures,
    )
    cost = compute_cost_usd(
        token_usage=parsed_response.token_usage,
        llm_calls=[call.model_dump() for call in parsed_response.llm_calls],
        pricing=config.pricing,
    )

    return CaseResult(
        run_id=run_id,
        case_id=case.case_id,
        category=case.category,
        scenario=case.scenario,
        query=case.query,
        session_id=session_id,
        endpoint=endpoint_url,
        upload_fixture=case.upload_fixture,
        request_payload=request_payload,
        request_id=parsed_response.request_id,
        http_status=parsed_response.http_status,
        response_text=parsed_response.response_text,
        response=response,
        debug=parsed_response.debug,
        answer_provenance=parsed_response.answer_provenance,
        evidence_assessment=evidence_scope,
        observed_hits=parsed_response.observed_hits,
        retrieval_diagnostics=parsed_response.retrieval_diagnostics,
        planner_diagnostics=parsed_response.planner_diagnostics,
        trace=parsed_response.response_trace,
        latency_ms_e2e=latency_ms_e2e,
        latency_ms_server=parsed_response.latency_ms_server,
        latency_breakdown=parsed_response.latency_breakdown,
        tool_calls=parsed_response.tool_calls,
        tool_call_count=parsed_response.tool_call_count,
        token_usage=parsed_response.token_usage,
        output_tokens=output_tokens,
        model_name=parsed_response.model_name,
        models_used=parsed_response.models_used,
        model_usage_status=parsed_response.model_usage_status,
        llm_calls=parsed_response.llm_calls,
        planner_errors=parsed_response.planner_errors,
        error_codes=parsed_response.error_codes,
        validation_events=parsed_response.validation_events,
        edge_decisions=parsed_response.edge_decisions,
        debug_errors=parsed_response.debug_errors,
        runtime_errors=runtime_errors,
        response_errors=response_errors,
        judge_errors=judge_errors,
        judge_audit_failures=judge_audit_failures,
        actions=parsed_response.actions,
        slack_delivery_status=slack_delivery_status,
        slack_delivery_required=slack_delivery_required,
        slack_delivery_error=slack_delivery_error,
        validator_reason=parsed_response.validator_reason,
        validator_feedback=validator_feedback,
        effective_weights=effective_weights.as_dict(),
        rule_scores=rule_scores,
        rule_score_total=rule_weighted,
        debug_schema_version=parsed_response.debug_schema_version,
        debug_observability_status=parsed_response.debug_observability_status,
        missing_required_debug_fields=parsed_response.missing_required_debug_fields,
        judge_subscores=judge_subscores,
        judge_score_total=llm_judge_score,
        llm_judge_score=llm_judge_score,
        llm_judge_reason=llm_judge_reason,
        judge_min_score_applied=judge_min_score,
        judge_input_complete=judge_input_complete,
        judge_gate_passed=judge_gate_passed,
        invalid_eval=any(str(error).startswith("invalid_eval:") for error in judge_errors),
        resolved_unit_count=resolved_unit_count,
        missing_reference_unit_count=missing_reference_unit_count,
        unchecked_unit_count=unchecked_unit_count,
        block_count=block_count,
        exact_match_unit_count=exact_match_unit_count,
        unsupported_unit_count=unsupported_unit_count,
        synthesis_mode=parsed_response.synthesis_mode,
        gate_failures=gate_failures,
        composite_quality_score=composite_quality_score,
        product_pass=product_pass,
        judge_pass=judge_pass,
        release_pass=release_pass,
        final_score=composite_quality_score,
        passed=release_pass,
        cost_usd=cost,
        created_at_utc=created_at,
    )


__all__ = [
    "build_case_result",
]
