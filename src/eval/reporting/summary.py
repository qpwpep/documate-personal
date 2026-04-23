from __future__ import annotations

from datetime import datetime, timezone

from ..config_models import BenchmarkCase, BenchmarkConfig
from ..metric_rules import tool_confusion_counts
from ..result_models import CaseResult
from ..summary_models import GateResult, RunSummary, RunTrack, SummaryStats
from .histograms import build_analysis, build_failure_reason, percentile


_AUDIT_DETERMINISTIC_DIRECT_USAGE_CEILING = 0.35
_AUDIT_HIGH_RULE_LOW_JUDGE_DIVERGENCE_CEILING = 0.10
_HIGH_RULE_LOW_JUDGE_DIVERGENCE_MARGIN = 0.35


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
    attempt_count = 0
    for result in results:
        if any(call.stage == "planner" for call in result.llm_calls):
            attempt_count += 1
            continue
        if result.planner_errors:
            attempt_count += 1
            continue
        if result.planner_diagnostics is not None and str(result.planner_diagnostics.status or "") not in {"deterministic", "missing"}:
            attempt_count += 1
    return attempt_count


def _compute_planner_structured_success_rate(results: list[CaseResult]) -> float:
    eligible = [
        result
        for result in _structured_success_cases(results)
        if result.planner_diagnostics is None or str(result.planner_diagnostics.status or "") != "deterministic"
    ]
    if not eligible:
        return 1.0
    successes = sum(
        1
        for result in eligible
        if result.planner_diagnostics is not None and str(result.planner_diagnostics.status or "") == "llm"
    )
    return round(successes / len(eligible), 4)


def _compute_synthesis_structured_success_rate(results: list[CaseResult]) -> float:
    eligible = _structured_success_cases(results)
    if not eligible:
        return 1.0
    successes = sum(1 for result in eligible if result.synthesis_mode == "structured_only")
    return round(successes / len(eligible), 4)


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
) -> RunSummary:
    case_map = {case.case_id: case for case in cases}
    scored_results = [result for result in results if result.composite_quality_score is not None]
    product_results = [result for result in scored_results if bool(result.product_pass)]
    release_results = [result for result in scored_results if bool(result.release_pass)]
    judge_eligible_results = [result for result in results if result.judge_pass is not None]
    judge_results = [result for result in judge_eligible_results if bool(result.judge_pass)]
    product_pass_rate = (len(product_results) / len(scored_results)) if scored_results else 0.0
    release_pass_rate = (len(release_results) / len(scored_results)) if scored_results else 0.0
    judge_pass_rate = (len(judge_results) / len(judge_eligible_results)) if judge_eligible_results else None

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
    llm_call_coverage_rate = sum(1 for result in results if result.llm_calls) / len(results) if results else 0.0
    request_id_coverage_rate = sum(1 for result in results if result.request_id) / len(results) if results else 0.0
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
        scored_cases=len(scored_results),
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
        synthesis_structured_success_rate=_compute_synthesis_structured_success_rate(results),
        failures=failures[:50],
    )
    analysis = build_analysis(case_map=case_map, results=results)
    hard_gates = config.hard_gates
    gates = [
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
        GateResult(name="judge_min_score_pass_rate", threshold=1.0, actual=metrics.judge_pass_rate, passed=metrics.judge_pass_rate is None or metrics.judge_pass_rate >= 1.0, gate_type="audit"),
        GateResult(name="judge_input_completeness_rate", threshold=1.0, actual=metrics.judge_input_completeness_rate, passed=metrics.judge_input_completeness_rate is None or metrics.judge_input_completeness_rate >= 1.0, gate_type="audit"),
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
            "judge_min_score_pass_rate": metrics.judge_pass_rate,
            "judge_min_score_failures": metrics.judge_min_score_failures,
            "llm_call_coverage_rate": metrics.llm_call_coverage_rate,
            "request_id_coverage_rate": metrics.request_id_coverage_rate,
            "judge_input_completeness_rate": metrics.judge_input_completeness_rate,
            "deterministic_direct_usage_rate": metrics.deterministic_direct_usage_rate,
            "high_rule_low_judge_divergence_rate": metrics.high_rule_low_judge_divergence_rate,
            "slack_delivery_success_rate": metrics.slack_delivery_success_rate,
            "slack_delivery_required_cases": metrics.slack_delivery_required_cases,
            "slack_delivery_success_cases": metrics.slack_delivery_success_cases,
            "cost_gate_eligible": metrics.cost_gate_eligible,
        },
    )


__all__ = [
    "build_summary",
]
