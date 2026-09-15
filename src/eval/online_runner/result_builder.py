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


_PRODUCT_PASS_FLOOR = 0.75


def _resolve_judge_min_score(case: BenchmarkCase, config: BenchmarkConfig) -> float | None:
    if case.judge_min_score is not None:
        return float(case.judge_min_score)
    return config.judge_min_score.for_category(case.category)


def _groundedness_gate_applies(case: BenchmarkCase) -> bool:
    """Evidence-based answers need groundedness; pure action cases do not."""
    return (
        case.category != "tool_action"
        or case.require_official_citation
        or case.require_local_citation
    )


def _build_gate_failures(
    *,
    runtime_errors: list[str],
    response_errors: list[str],
    debug_errors: list[str],
    missing_required_debug_fields: list[str],
    product_pass: bool | None,
    judge_pass: bool | None,
    judge_status: str | None,
    judge_status_reason: str | None,
    eval_validity: str | None,
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
    if judge_status == "disabled":
        failures.append("judge_disabled")
    elif judge_status == "failed":
        failures.append("judge_failed")
    elif judge_status == "not_run":
        failures.append(
            "judge_input_incomplete" if judge_status_reason == "input_incomplete" else "judge_not_run"
        )
    if judge_audit_failures and "judge_min_score_audit_failed" not in failures:
        failures.append("judge_min_score_audit_failed")
    if eval_validity == "invalid" and "judge_failed" not in failures:
        failures.append("invalid_eval")
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
    judge_input_issues: list[str] = []
    judge_status: str | None = None
    judge_status_reason: str | None = None
    llm_judge_score: float | None = None
    llm_judge_reason: str | None = None
    judge_subscores: JudgeSubscores | None = None
    judge_input_complete: bool | None = None
    judge_required = bool(config.judge_enabled)
    judge_min_score = _resolve_judge_min_score(case, config)
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
    has_final_response = response is not None and bool(parsed_response.response_text.strip())
    if not judge_required:
        judge_status = "disabled"
    elif not has_final_response:
        # A product failure already settles the verdict; this is not a judge outage.
        judge_status = "not_run"
        judge_status_reason = "missing_final_response"
    else:
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
        judge_input_issues = judge.payload_completeness_issues(judge_payload)
        judge_input_complete = not judge_input_issues
        outcome = judge.score_case(
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
        if outcome.status == "disabled":
            # Config requires the judge but the client was built disabled or is missing.
            judge_status = "failed"
            judge_status_reason = "client_unavailable"
            judge_errors.append(outcome.error or "invalid_eval: judge client is not initialized")
        else:
            judge_status = outcome.status
            judge_status_reason = outcome.failure_kind
            if outcome.error:
                judge_errors.append(outcome.error)
            if outcome.input_issues:
                judge_input_issues = outcome.input_issues
                judge_input_complete = False
            llm_judge_score = outcome.score
            llm_judge_reason = outcome.reason
            judge_subscores = outcome.subscores

    if judge_status == "succeeded":
        if judge_min_score is None:
            judge_audit_failures.append(
                f"judge_min_score audit failed: no threshold configured for category '{case.category}'"
            )
        elif llm_judge_score is not None and llm_judge_score < judge_min_score:
            judge_audit_failures.append(
                "judge_min_score audit failed: "
                f"score={llm_judge_score:.3f} threshold={judge_min_score:.3f}"
            )
        if judge_subscores is not None:
            subscore_minimums = {"answer_quality": config.judge_min_subscores.answer_quality}
            if _groundedness_gate_applies(case):
                subscore_minimums["groundedness"] = config.judge_min_subscores.groundedness
            for name, minimum in subscore_minimums.items():
                if minimum is None:
                    continue
                value = float(getattr(judge_subscores, name))
                if value < minimum:
                    judge_audit_failures.append(
                        f"judge_min_subscore audit failed: {name}={value:.3f} threshold={minimum:.3f}"
                    )
        judge_pass = not judge_audit_failures
    else:
        judge_pass = None

    rule_scores = compute_rule_scores(
        case=case,
        response=response,
        called_tools=parsed_response.tool_calls,
        observed_hits=parsed_response.observed_hits,
        runtime_errors=runtime_errors,
        response_errors=response_errors,
        validator_reason=parsed_response.validator_reason,
        synthesis_mode=parsed_response.synthesis_mode,
        slack_delivery_required=slack_delivery_required,
        slack_delivery_status=slack_delivery_status,
        evidence_scope=evidence_scope,
    )
    rule_weighted = compute_rule_weighted_score(rule_scores, effective_weights)

    composite_quality_score = compute_composite_quality_score(
        rule_weighted_score=rule_weighted,
        llm_judge_score=llm_judge_score if judge_status == "succeeded" else None,
        weights=effective_weights,
    )
    if composite_quality_score is not None:
        product_pass = composite_quality_score >= _PRODUCT_PASS_FLOOR
    elif runtime_errors or response_errors:
        product_pass = False
    else:
        product_pass = None

    if judge_status == "failed":
        eval_validity = "invalid"
    elif judge_status == "disabled":
        eval_validity = "incomplete"
    elif judge_status == "not_run":
        # missing_final_response is a settled product verdict, not an
        # evaluation gap. Any other unrun reason leaves the case incomplete.
        eval_validity = "valid" if judge_status_reason == "missing_final_response" else "incomplete"
    else:
        eval_validity = "valid"

    release_pass = bool(
        eval_validity == "valid"
        and product_pass is True
        and judge_pass is True
        and not runtime_errors
        and not response_errors
    )
    gate_failures = _build_gate_failures(
        runtime_errors=runtime_errors,
        response_errors=response_errors,
        debug_errors=parsed_response.debug_errors,
        missing_required_debug_fields=parsed_response.missing_required_debug_fields,
        product_pass=product_pass,
        judge_pass=judge_pass,
        judge_status=judge_status,
        judge_status_reason=judge_status_reason,
        eval_validity=eval_validity,
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
        upload_fixtures=case.resolved_upload_fixtures,
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
        judge_status=judge_status,
        judge_status_reason=judge_status_reason,
        eval_validity=eval_validity,
        judge_input_issues=judge_input_issues,
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
        judge_min_score_applied=judge_min_score if judge_required else None,
        judge_input_complete=judge_input_complete,
        invalid_eval=(eval_validity == "invalid"),
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
