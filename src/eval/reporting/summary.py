from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from src.core.planner_schema import PLANNER_WARNING_DUPLICATE_ROUTE_MERGED

from ..config_models import BenchmarkCase, BenchmarkConfig
from ..metric_rules import tool_confusion_counts
from ..result_models import CaseResult
from ..summary_models import GateResult, RunSummary, RunTrack, SummaryStats
from .histograms import build_analysis, build_failure_reason, percentile
from .latency_values import result_latency_breakdown_stage_ms, result_server_latency_ms


_AUDIT_DETERMINISTIC_DIRECT_USAGE_CEILING = 0.35
_AUDIT_HIGH_RULE_LOW_JUDGE_DIVERGENCE_CEILING = 0.10
_HIGH_RULE_LOW_JUDGE_DIVERGENCE_MARGIN = 0.35
EXECUTION_CONTRACT_VERSION = "shared-client-scenario-v1"
MEASUREMENT_CONTRACT_VERSION = "attachment-question-scenario-v1"
# v2: judge outcome is a hard release gate; missing scores no longer
# renormalize into a composite; every category has explicit judge minimums.
SCORING_CONTRACT_VERSION = "judge-state-contract-v2"


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scenario_latency_metrics(results: list[CaseResult]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for field in ("attachment_setup_ms", "question_response_ms", "scenario_total_ms"):
        values = [int(value) for result in results if (value := getattr(result, field)) is not None]
        for name, quantile in (("p50", 0.50), ("p95", 0.95)):
            value = percentile(values, quantile)
            metrics[f"{name}_{field}"] = round(value, 2) if value is not None else None
    return metrics


def _call_has_usage(call: Any) -> bool:
    if not isinstance(call, dict):
        return False
    response_metadata = call.get("response_metadata")
    candidates = [call.get("usage_metadata")]
    if isinstance(response_metadata, dict):
        candidates.append(response_metadata.get("token_usage"))
    for usage in candidates:
        if not isinstance(usage, dict):
            continue
        prompt = usage.get("input_tokens", usage.get("prompt_tokens"))
        completion = usage.get("output_tokens", usage.get("completion_tokens"))
        if isinstance(prompt, int) and not isinstance(prompt, bool) and prompt >= 0:
            if isinstance(completion, int) and not isinstance(completion, bool) and completion >= 0:
                return True
    return False


def _has_llm_coverage(result: CaseResult) -> bool:
    if not result.scenario_turns:
        return bool(result.llm_calls)
    has_calls = False
    for turn in result.scenario_turns:
        debug = turn.debug
        if not debug or debug.get("missing_required_debug_fields"):
            return False
        calls = debug.get("llm_calls")
        if debug.get("model_usage_status") == "deterministic" and calls == []:
            continue
        if not isinstance(calls, list) or not calls or not all(_call_has_usage(call) for call in calls):
            return False
        has_calls = True
    return has_calls


def _has_request_id_coverage(result: CaseResult) -> bool:
    if result.scenario_turns:
        return all(bool(turn.request_id) for turn in result.scenario_turns)
    return bool(result.request_id)


def _structured_success_cases(results: list[CaseResult]) -> list[CaseResult]:
    return [result for result in results if result.category in {"docs_only", "rag_only", "hybrid"}]


def _compute_planner_deterministic_rate(results: list[CaseResult]) -> float:
    if not results:
        return 1.0
    deterministic_count = sum(
        1
        for result in results
        if result.planner_diagnostics is not None and str(result.planner_diagnostics.status or "") == "deterministic"
    )
    return round(deterministic_count / len(results), 4)


def _compute_planner_llm_attempt_count(results: list[CaseResult]) -> int:
    return sum(1 for result in results if _planner_llm_attempted(result))


def _planner_llm_attempted(result: CaseResult) -> bool:
    if any(call.stage == "planner" for call in result.llm_calls):
        return True
    if result.planner_errors:
        return True
    if result.planner_diagnostics is None:
        return False
    return str(result.planner_diagnostics.status or "") not in {"deterministic", "missing"}


def _compute_planner_structured_success_rate(results: list[CaseResult]) -> float:
    eligible = [
        result
        for result in _structured_success_cases(results)
        if _planner_llm_attempted(result)
    ]
    if not eligible:
        return 1.0
    successes = sum(
        1
        for result in eligible
        if result.planner_diagnostics is not None
        and str(result.planner_diagnostics.status or "") == "llm"
        and not result.planner_errors
    )
    return round(successes / len(eligible), 4)


def _planner_warnings(result: CaseResult) -> list[str]:
    if result.planner_diagnostics is None:
        return []
    return [
        str(warning).strip()
        for warning in result.planner_diagnostics.planner_warnings
        if str(warning).strip()
    ]


def _compute_planner_warning_count(results: list[CaseResult]) -> int:
    return sum(len(_planner_warnings(result)) for result in results)


def _compute_planner_duplicate_route_merge_count(results: list[CaseResult]) -> int:
    return sum(
        1
        for result in results
        if PLANNER_WARNING_DUPLICATE_ROUTE_MERGED in set(_planner_warnings(result))
    )


def _compute_planner_final_success_rate(results: list[CaseResult]) -> float:
    eligible = _structured_success_cases(results)
    if not eligible:
        return 1.0
    successes = sum(
        1
        for result in eligible
        if result.planner_diagnostics is not None
        and str(result.planner_diagnostics.status or "") in {"llm", "deterministic", "heuristic_fallback"}
        and not result.planner_errors
    )
    return round(successes / len(eligible), 4)


def _compute_synthesis_structured_success_rate(results: list[CaseResult]) -> float:
    eligible = _structured_success_cases(results)
    if not eligible:
        return 1.0
    successes = sum(1 for result in eligible if result.synthesis_mode == "structured_only")
    return round(successes / len(eligible), 4)


def _rounded_p95(values: list[int]) -> float | None:
    value = percentile(values, 0.95)
    return round(value, 2) if value is not None else None


def _category_p95_metric(
    results: list[CaseResult],
    category: str,
    value_getter: Callable[[CaseResult], int | None],
) -> float | None:
    values: list[int] = []
    for result in results:
        if result.category != category:
            continue
        value = value_getter(result)
        if value is not None:
            values.append(int(value))
    return _rounded_p95(values)


def _result_e2e_latency_ms(result: CaseResult) -> int | None:
    return result.latency_ms_e2e


def _category_p95_latency(results: list[CaseResult], category: str) -> float | None:
    return _category_p95_metric(results, category, _result_e2e_latency_ms)


def build_summary(
    *,
    run_id: str,
    endpoint: str,
    fixtures_path: str,
    config_path: str,
    track: RunTrack,
    requested_limit: int | None,
    config: BenchmarkConfig,
    cases: list[BenchmarkCase],
    results: list[CaseResult],
    slack_live_enabled: bool = False,
    attachment_fingerprints: dict[str, str] | None = None,
    execution_options: dict[str, Any] | None = None,
) -> RunSummary:
    case_map = {case.case_id: case for case in cases}
    planned_ids = {case.case_id for case in cases}
    result_ids = [result.case_id for result in results]
    missing_result_cases = len(planned_ids.difference(result_ids))
    unexpected_result_cases = len(set(result_ids).difference(planned_ids))
    duplicate_result_cases = len(result_ids) - len(set(result_ids))

    # Denominators never shrink to the cases that happened to score. Planned
    # cases are the release denominator; scored verdicts are the product/judge
    # denominators.
    planned_cases = len(cases)
    valid_verdicts = [
        result
        for result in results
        if result.eval_validity == "valid"
        or (result.eval_validity in {None, "legacy_unknown"} and result.composite_quality_score is not None)
    ]
    invalid_eval_cases = sum(1 for result in results if result.eval_validity == "invalid")
    incomplete_eval_cases = sum(1 for result in results if result.eval_validity == "incomplete")
    product_results = [result for result in valid_verdicts if bool(result.product_pass)]
    release_results = [result for result in results if bool(result.release_pass)]
    judge_eligible_results = [result for result in results if result.judge_pass is not None]
    judge_results = [result for result in judge_eligible_results if bool(result.judge_pass)]
    product_pass_rate = (len(product_results) / len(valid_verdicts)) if valid_verdicts else 0.0
    release_pass_rate = (len(release_results) / planned_cases) if planned_cases else 0.0
    judge_pass_rate = (len(judge_results) / len(judge_eligible_results)) if judge_eligible_results else None

    def _judge_bucket(result: CaseResult) -> str:
        status = result.judge_status
        if status in {None, "legacy_unknown"}:
            if result.llm_judge_score is not None:
                return "succeeded"
            if result.judge_errors:
                return "failed"
            return "legacy_unknown"
        return str(status)

    judge_buckets = [_judge_bucket(result) for result in results]
    judge_succeeded_cases = judge_buckets.count("succeeded")
    judge_failed_cases = judge_buckets.count("failed")
    judge_disabled_cases = judge_buckets.count("disabled")
    judge_not_run_cases = judge_buckets.count("not_run")
    judge_legacy_unknown_cases = judge_buckets.count("legacy_unknown")
    judge_required_cases = planned_cases if config.judge_enabled else 0
    judge_execution_rate = (
        judge_succeeded_cases / judge_required_cases if judge_required_cases else None
    )

    tp_total = fp_total = fn_total = 0
    for result in results:
        case = case_map.get(result.case_id)
        if not case:
            continue
        tp, fp, fn = tool_confusion_counts(case=case, called_tools=result.tool_calls)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    tool_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 1.0
    tool_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 1.0

    citation_scores = []
    for result in results:
        case = case_map.get(result.case_id)
        if case and (case.require_official_citation or case.require_local_citation):
            citation_scores.append(float(result.rule_scores.get("citation_traceability", 0.0)))
    citation_compliance = (sum(citation_scores) / len(citation_scores)) if citation_scores else 1.0

    latencies = [int(result.latency_ms_e2e) for result in results if result.latency_ms_e2e is not None]
    p50_latency = percentile(latencies, 0.50)
    p95_latency = percentile(latencies, 0.95)
    hybrid_p95_latency = _category_p95_latency(results, "hybrid")
    hybrid_p95_server = _category_p95_metric(results, "hybrid", result_server_latency_ms)
    hybrid_p95_synthesis = _category_p95_metric(
        results,
        "hybrid",
        lambda result: result_latency_breakdown_stage_ms(result, "synthesis_total_ms"),
    )
    hybrid_p95_retrieval = _category_p95_metric(
        results,
        "hybrid",
        lambda result: result_latency_breakdown_stage_ms(result, "retrieval_total_ms"),
    )
    docs_only_p95_latency = _category_p95_latency(results, "docs_only")
    cost_values = [float(result.cost_usd) for result in results if result.cost_usd is not None]
    avg_cost = (sum(cost_values) / len(cost_values)) if cost_values else None
    slack_delivery_required_results = [result for result in results if result.slack_delivery_required]
    slack_delivery_success_results = [
        result for result in slack_delivery_required_results if result.slack_delivery_status == "success"
    ]
    slack_delivery_success_rate = (
        len(slack_delivery_success_results) / len(slack_delivery_required_results)
        if slack_live_enabled and slack_delivery_required_results
        else None
    )
    llm_call_coverage_rate = sum(1 for result in results if _has_llm_coverage(result)) / len(results) if results else 0.0
    request_id_coverage_rate = sum(1 for result in results if _has_request_id_coverage(result)) / len(results) if results else 0.0
    judge_input_eligible = [result for result in results if result.judge_input_complete is not None]
    judge_input_completeness_rate = (
        sum(1 for result in judge_input_eligible if result.judge_input_complete) / len(judge_input_eligible)
        if judge_input_eligible
        else None
    )
    judge_min_score_failures = sum(1 for result in results if "judge_min_score_audit_failed" in result.gate_failures)
    deterministic_direct_usage_rate = (
        sum(1 for result in _structured_success_cases(results) if result.synthesis_mode == "deterministic_grounded_direct")
        / len(_structured_success_cases(results))
        if _structured_success_cases(results)
        else 0.0
    )
    high_rule_low_judge_divergence_rate = (
        sum(
            1
            for result in results
            if result.rule_score_total is not None
            and result.llm_judge_score is not None
            and float(result.rule_score_total) - float(result.llm_judge_score)
            >= _HIGH_RULE_LOW_JUDGE_DIVERGENCE_MARGIN
        )
        / len([result for result in results if result.rule_score_total is not None and result.llm_judge_score is not None])
        if [result for result in results if result.rule_score_total is not None and result.llm_judge_score is not None]
        else 0.0
    )
    cost_gate_eligible = llm_call_coverage_rate >= float(config.hard_gates.cost_gate_min_llm_call_coverage)

    failures = [
        {"case_id": result.case_id, "category": result.category, "reason": build_failure_reason(result)}
        for result in results
        if not result.release_pass
    ]

    metrics = SummaryStats(
        total_cases=len(results),
        planned_cases=planned_cases,
        scored_cases=len(valid_verdicts),
        invalid_eval_cases=invalid_eval_cases,
        incomplete_eval_cases=incomplete_eval_cases,
        missing_result_cases=missing_result_cases,
        duplicate_result_cases=duplicate_result_cases,
        unexpected_result_cases=unexpected_result_cases,
        judge_required_cases=judge_required_cases,
        judge_succeeded_cases=judge_succeeded_cases,
        judge_failed_cases=judge_failed_cases,
        judge_disabled_cases=judge_disabled_cases,
        judge_not_run_cases=judge_not_run_cases,
        judge_legacy_unknown_cases=judge_legacy_unknown_cases,
        judge_execution_rate=round(judge_execution_rate, 4) if judge_execution_rate is not None else None,
        passed_cases=len(release_results),
        pass_rate=round(release_pass_rate, 4),
        product_passed_cases=len(product_results),
        judge_passed_cases=len(judge_results),
        release_passed_cases=len(release_results),
        product_pass_rate=round(product_pass_rate, 4),
        judge_pass_rate=round(judge_pass_rate, 4) if judge_pass_rate is not None else None,
        release_pass_rate=round(release_pass_rate, 4),
        tool_precision=round(tool_precision, 4),
        tool_recall=round(tool_recall, 4),
        citation_compliance=round(citation_compliance, 4),
        p50_latency_ms=round(p50_latency, 2) if p50_latency is not None else None,
        p95_latency_ms=round(p95_latency, 2) if p95_latency is not None else None,
        **_scenario_latency_metrics(results),
        hybrid_p95_latency_ms=hybrid_p95_latency,
        hybrid_p95_server_ms=hybrid_p95_server,
        hybrid_p95_synthesis_ms=hybrid_p95_synthesis,
        hybrid_p95_retrieval_ms=hybrid_p95_retrieval,
        docs_only_p95_latency_ms=docs_only_p95_latency,
        avg_cost_per_case_usd=round(avg_cost, 8) if avg_cost is not None else None,
        slack_delivery_required_cases=len(slack_delivery_required_results),
        slack_delivery_success_cases=len(slack_delivery_success_results),
        slack_delivery_success_rate=round(slack_delivery_success_rate, 4) if slack_delivery_success_rate is not None else None,
        cost_gate_eligible=cost_gate_eligible,
        llm_call_coverage_rate=round(llm_call_coverage_rate, 4),
        request_id_coverage_rate=round(request_id_coverage_rate, 4),
        judge_input_completeness_rate=round(judge_input_completeness_rate, 4) if judge_input_completeness_rate is not None else None,
        judge_min_score_failures=judge_min_score_failures,
        deterministic_direct_usage_rate=round(deterministic_direct_usage_rate, 4),
        high_rule_low_judge_divergence_rate=round(high_rule_low_judge_divergence_rate, 4),
        planner_deterministic_rate=_compute_planner_deterministic_rate(results),
        planner_llm_attempt_count=_compute_planner_llm_attempt_count(results),
        planner_structured_success_rate=_compute_planner_structured_success_rate(results),
        planner_error_count=sum(len(result.planner_errors) for result in results),
        planner_error_case_count=sum(1 for result in results if result.planner_errors),
        planner_warning_count=_compute_planner_warning_count(results),
        planner_duplicate_route_merge_count=_compute_planner_duplicate_route_merge_count(results),
        planner_final_success_rate=_compute_planner_final_success_rate(results),
        synthesis_structured_success_rate=_compute_synthesis_structured_success_rate(results),
        failures=failures[:50],
    )
    analysis = build_analysis(case_map=case_map, results=results)
    hard_gates = config.hard_gates
    completeness_issues = (
        invalid_eval_cases
        + incomplete_eval_cases
        + missing_result_cases
        + duplicate_result_cases
        + unexpected_result_cases
    )
    gates = [
        GateResult(
            name="evaluation_completeness",
            threshold=0,
            actual=completeness_issues,
            passed=completeness_issues == 0,
            gate_type="release",
            detail=(
                f"invalid={invalid_eval_cases} incomplete={incomplete_eval_cases} "
                f"missing={missing_result_cases} duplicate={duplicate_result_cases} "
                f"unexpected={unexpected_result_cases}"
            ),
        ),
        GateResult(name="release_pass_rate", threshold=hard_gates.pass_rate, actual=metrics.release_pass_rate, passed=metrics.release_pass_rate >= hard_gates.pass_rate, gate_type="release"),
        GateResult(name="tool_precision", threshold=hard_gates.tool_precision, actual=metrics.tool_precision, passed=metrics.tool_precision >= hard_gates.tool_precision, gate_type="release"),
        GateResult(name="tool_recall", threshold=hard_gates.tool_recall, actual=metrics.tool_recall, passed=metrics.tool_recall >= hard_gates.tool_recall, gate_type="release"),
        GateResult(name="citation_compliance", threshold=hard_gates.citation_compliance, actual=metrics.citation_compliance, passed=metrics.citation_compliance >= hard_gates.citation_compliance, gate_type="release"),
        GateResult(name="p95_latency_ms", threshold=hard_gates.p95_latency_ms, actual=metrics.p95_latency_ms, passed=metrics.p95_latency_ms is not None and metrics.p95_latency_ms <= hard_gates.p95_latency_ms, gate_type="release"),
        GateResult(
            name="avg_cost_per_case_usd",
            threshold=hard_gates.avg_cost_per_case_usd,
            actual=metrics.avg_cost_per_case_usd if metrics.cost_gate_eligible else None,
            passed=True if not metrics.cost_gate_eligible else (metrics.avg_cost_per_case_usd is not None and metrics.avg_cost_per_case_usd <= hard_gates.avg_cost_per_case_usd),
            gate_type="release",
            status="evaluated" if metrics.cost_gate_eligible else "skipped_insufficient_coverage",
        ),
        GateResult(
            name="judge_min_score_pass_rate",
            threshold=1.0,
            actual=metrics.judge_pass_rate,
            # A required judge that never produced a verdict is a failure, not a skip.
            passed=(
                metrics.judge_pass_rate is not None and metrics.judge_pass_rate >= 1.0
                if metrics.judge_required_cases
                else metrics.judge_pass_rate is None or metrics.judge_pass_rate >= 1.0
            ),
            gate_type="audit",
        ),
        GateResult(
            name="judge_input_completeness_rate",
            threshold=1.0,
            actual=metrics.judge_input_completeness_rate,
            passed=(
                metrics.judge_input_completeness_rate is not None and metrics.judge_input_completeness_rate >= 1.0
                if metrics.judge_required_cases
                else metrics.judge_input_completeness_rate is None or metrics.judge_input_completeness_rate >= 1.0
            ),
            gate_type="audit",
        ),
        GateResult(name="deterministic_direct_usage_rate", threshold=_AUDIT_DETERMINISTIC_DIRECT_USAGE_CEILING, actual=metrics.deterministic_direct_usage_rate, passed=metrics.deterministic_direct_usage_rate <= _AUDIT_DETERMINISTIC_DIRECT_USAGE_CEILING, gate_type="audit"),
        GateResult(name="high_rule_low_judge_divergence_rate", threshold=_AUDIT_HIGH_RULE_LOW_JUDGE_DIVERGENCE_CEILING, actual=metrics.high_rule_low_judge_divergence_rate, passed=metrics.high_rule_low_judge_divergence_rate <= _AUDIT_HIGH_RULE_LOW_JUDGE_DIVERGENCE_CEILING, gate_type="audit"),
        GateResult(
            name="slack_delivery_success_rate",
            threshold=1.0,
            actual=metrics.slack_delivery_success_rate,
            passed=(
                True
                if not slack_live_enabled or not slack_delivery_required_results
                else (metrics.slack_delivery_success_rate is not None and metrics.slack_delivery_success_rate >= 1.0)
            ),
            gate_type="audit",
            status=(
                "skipped_not_live"
                if not slack_live_enabled
                else ("skipped_no_required_cases" if not slack_delivery_required_results else "evaluated")
            ),
        ),
    ]
    overall_passed = all(gate.passed for gate in gates if gate.gate_type == "release")
    return RunSummary(
        run_id=run_id,
        endpoint=endpoint,
        fixtures_path=fixtures_path,
        config_path=config_path,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        mode="online",
        track=track,
        requested_limit=requested_limit,
        execution_contract_version=EXECUTION_CONTRACT_VERSION,
        measurement_contract_version=MEASUREMENT_CONTRACT_VERSION,
        suite_fingerprint=_fingerprint({
            "cases": [case.model_dump(mode="json") for case in cases],
            "attachments": attachment_fingerprints or {},
        }),
        evaluation_fingerprint=_fingerprint({
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "config": config.model_dump(mode="json"),
            "slack_live_enabled": slack_live_enabled,
            "execution_options": execution_options or {},
        }),
        metrics=metrics,
        analysis=analysis,
        gates=gates,
        overall_passed=overall_passed,
        weights=config.weights.as_dict(),
        hard_gates=config.hard_gates.model_dump(),
        pricing=config.pricing.model_dump(),
        judge_enabled=config.judge_enabled,
        judge_model=config.judge_model,
        audit_metrics={
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "judge_min_score_pass_rate": metrics.judge_pass_rate,
            "judge_min_score_failures": metrics.judge_min_score_failures,
            "judge_required_cases": metrics.judge_required_cases,
            "judge_succeeded_cases": metrics.judge_succeeded_cases,
            "judge_failed_cases": metrics.judge_failed_cases,
            "judge_not_run_cases": metrics.judge_not_run_cases,
            "judge_disabled_cases": metrics.judge_disabled_cases,
            "judge_execution_rate": metrics.judge_execution_rate,
            "invalid_eval_cases": metrics.invalid_eval_cases,
            "incomplete_eval_cases": metrics.incomplete_eval_cases,
            "llm_call_coverage_rate": metrics.llm_call_coverage_rate,
            "request_id_coverage_rate": metrics.request_id_coverage_rate,
            "judge_input_completeness_rate": metrics.judge_input_completeness_rate,
            "deterministic_direct_usage_rate": metrics.deterministic_direct_usage_rate,
            "high_rule_low_judge_divergence_rate": metrics.high_rule_low_judge_divergence_rate,
            "planner_error_count": metrics.planner_error_count,
            "planner_error_case_count": metrics.planner_error_case_count,
            "planner_warning_count": metrics.planner_warning_count,
            "planner_duplicate_route_merge_count": metrics.planner_duplicate_route_merge_count,
            "planner_final_success_rate": metrics.planner_final_success_rate,
            "slack_delivery_success_rate": metrics.slack_delivery_success_rate,
            "slack_delivery_required_cases": metrics.slack_delivery_required_cases,
            "slack_delivery_success_cases": metrics.slack_delivery_success_cases,
            "cost_gate_eligible": metrics.cost_gate_eligible,
        },
    )


__all__ = [
    "build_summary",
]
