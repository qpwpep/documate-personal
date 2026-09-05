from __future__ import annotations

from src.core.contracts.debug import ActionResults
from src.core.answer_schema.rendering import filter_claims_by_evidence
from ..judge_llm import LLMJudge
from ..metric_rules import compute_rule_scores
from ..pricing import compute_cost_usd
from ..config_models import BenchmarkCase, BenchmarkConfig
from ..result_models import CaseResult, JudgeSubscores
from ..weighting import (
    compute_composite_quality_score,
    compute_rule_weighted_score,
    resolve_base_weights_for_case,
    resolve_effective_weights,
)
from .response_parser import ParsedResponseData, parse_claims_from_response_payload, parse_sections_from_response_payload


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
    action_results: ActionResults | None,
) -> tuple[str, str | None]:
    if not slack_delivery_required:
        return "not_applicable", None
    if action_results is None or action_results.slack_notify is None:
        return "unknown", "missing_action_results"

    slack_result = action_results.slack_notify
    raw_status = str(slack_result.status or "").strip().lower()
    if raw_status in {"ok", "success"}:
        return "success", None
    if raw_status == "skipped":
        return "skipped", slack_result.reason or slack_result.error
    if raw_status in {"error", "failed"}:
        return "failed", slack_result.error or slack_result.reason
    return "unknown", slack_result.error or slack_result.reason or raw_status or "unknown_status"


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
    if weights_error:
        runtime_errors.append(f"weight_override error: {weights_error}")

    response_payload = parsed_response.response_payload
    response_claims = parse_claims_from_response_payload(response_payload)
    response_sections = parse_sections_from_response_payload(response_payload)
    section_count = len(response_sections)
    output_tokens = _extract_output_tokens(parsed_response)
    valid_claim_count = 0
    invalid_claim_count = 0
    if response_claims:
        valid_claims, invalid_claims = filter_claims_by_evidence(
            claims=response_claims,
            evidence_items=parsed_response.observed_evidence or parsed_response.response_evidence,
        )
        valid_claim_count = len(valid_claims)
        invalid_claim_count = len(invalid_claims)
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
        action_results=parsed_response.action_results,
    )

    judge_errors: list[str] = []
    judge_audit_failures: list[str] = []
    llm_judge_score: float | None = None
    llm_judge_reason: str | None = None
    judge_subscores: JudgeSubscores | None = None
    judge_input_complete: bool | None = None
    if parsed_response.response_text.strip() and config.judge_enabled:
        judge_payload = judge.build_case_payload(
            case=case,
            response_text=parsed_response.response_text,
            tool_calls=parsed_response.tool_calls,
            claims=response_claims,
            response_evidence=parsed_response.response_evidence,
            sections=response_sections,
            observed_evidence=parsed_response.observed_evidence,
            retrieval_diagnostics=parsed_response.retrieval_diagnostics,
            planner_diagnostics=parsed_response.planner_diagnostics,
            validator_reason=parsed_response.validator_reason,
            synthesis_mode=parsed_response.synthesis_mode,
            valid_claim_count=valid_claim_count,
            invalid_claim_count=invalid_claim_count,
            tool_call_count=parsed_response.tool_call_count,
            action_results=parsed_response.action_results,
            slack_delivery_required=slack_delivery_required,
        )
        judge_input_complete = judge.is_payload_complete(judge_payload)
        llm_judge_score, llm_judge_reason, judge_error, judge_subscores = judge.score_case(
            case=case,
            response_text=parsed_response.response_text,
            tool_calls=parsed_response.tool_calls,
            claims=response_claims,
            response_evidence=parsed_response.response_evidence,
            sections=response_sections,
            observed_evidence=parsed_response.observed_evidence,
            retrieval_diagnostics=parsed_response.retrieval_diagnostics,
            planner_diagnostics=parsed_response.planner_diagnostics,
            validator_reason=parsed_response.validator_reason,
            synthesis_mode=parsed_response.synthesis_mode,
            valid_claim_count=valid_claim_count,
            invalid_claim_count=invalid_claim_count,
            tool_call_count=parsed_response.tool_call_count,
            action_results=parsed_response.action_results,
            slack_delivery_required=slack_delivery_required,
        )
        if judge_error:
            judge_errors.append(judge_error)

    rule_scores = compute_rule_scores(
        case=case,
        response_text=parsed_response.response_text,
        called_tools=parsed_response.tool_calls,
        response_evidence=parsed_response.response_evidence,
        observed_evidence=parsed_response.observed_evidence,
        runtime_errors=runtime_errors,
        response_errors=parsed_response.response_errors,
        judge_errors=judge_errors,
        validator_reason=parsed_response.validator_reason,
        response_claims=response_claims,
        synthesis_mode=parsed_response.synthesis_mode,
        invalid_claim_count=invalid_claim_count,
        response_sections=response_sections,
        slack_delivery_required=slack_delivery_required,
        slack_delivery_status=slack_delivery_status,
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
    release_pass = bool(product_pass and not runtime_errors and not parsed_response.response_errors)
    gate_failures = _build_gate_failures(
        runtime_errors=runtime_errors,
        response_errors=parsed_response.response_errors,
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
        response_payload=response_payload,
        response_claims=response_claims,
        evidence=parsed_response.response_evidence,
        observed_evidence=parsed_response.observed_evidence,
        retrieval_diagnostics=parsed_response.retrieval_diagnostics,
        planner_diagnostics=parsed_response.planner_diagnostics,
        file_path=parsed_response.response_file_path,
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
        response_errors=parsed_response.response_errors,
        judge_errors=judge_errors,
        judge_audit_failures=judge_audit_failures,
        action_results=parsed_response.action_results,
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
        valid_claim_count=valid_claim_count,
        invalid_claim_count=invalid_claim_count,
        section_count=section_count,
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
