from __future__ import annotations

from collections import Counter, defaultdict
import json
from datetime import datetime, timezone
from pathlib import Path

from ...contracts.routes import ROUTE_ORDER, route_for_tool, sort_routes
from ..scoring_rules import tool_confusion_counts
from ..schemas import (
    AnalysisStats,
    BenchmarkCase,
    BenchmarkConfig,
    CaseResult,
    CategoryPassRate,
    GateResult,
    LatencyBreakdownCoverage,
    PlannerDiagnosticsBucket,
    PlannerErrorBucket,
    RetrievalRouteStatusBucket,
    RetrievalWarningBucket,
    RouteConfusionBucket,
    RunSummary,
    StageLatencyPercentile,
    SummaryStats,
    SynthesisModeBucket,
    ValidatorReasonBucket,
    dump_jsonl,
)

_CATEGORY_ORDER: tuple[str, ...] = ("docs_only", "rag_only", "hybrid", "tool_action")
_PLANNER_ERROR_ORDER: tuple[str, ...] = (
    "structured_output_invocation_failed",
    "output_validation_failed",
    "sanitized_output_validation_failed",
    "upload_route_dropped",
)
_LATENCY_STAGE_FIELDS: tuple[str, ...] = (
    "upload_retriever_build_ms",
    "summarize_ms",
    "planner_ms",
    "retrieval_total_ms",
    "synthesis_total_ms",
    "validation_ms",
    "action_postprocess_ms",
)
_AUDIT_DETERMINISTIC_DIRECT_USAGE_CEILING = 0.35
_AUDIT_HIGH_RULE_LOW_JUDGE_DIVERGENCE_CEILING = 0.10
_HIGH_RULE_LOW_JUDGE_DIVERGENCE_MARGIN = 0.35


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = rank - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac)


def _category_sort_key(value: str) -> tuple[int, str]:
    if value in _CATEGORY_ORDER:
        return (_CATEGORY_ORDER.index(value), value)
    return (len(_CATEGORY_ORDER), value)


def _route_sort_key(value: str) -> tuple[int, str]:
    if value in ROUTE_ORDER:
        return (ROUTE_ORDER.index(value), value)
    return (len(ROUTE_ORDER), value)


def _planner_error_sort_key(value: str) -> tuple[int, str]:
    if value in _PLANNER_ERROR_ORDER:
        return (_PLANNER_ERROR_ORDER.index(value), value)
    return (len(_PLANNER_ERROR_ORDER), value)


def _sort_categories(values: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=_category_sort_key)


def _sort_routes(values: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    return sort_routes(values)


def _normalize_reason_text(text: str | None, *, max_length: int = 160) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return ""
    if len(normalized) <= max_length:
        return normalized
    truncated = normalized[: max_length - 3].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.rstrip(" .,:;") + "..."


def _build_rule_score_signature(result: CaseResult) -> str:
    low_scores = [
        (str(name), float(value))
        for name, value in result.rule_scores.items()
        if float(value) < 1.0
    ]
    if result.judge_subscores is not None:
        low_scores.extend(
            [
                (f"judge.{name}", float(value))
                for name, value in result.judge_subscores.model_dump().items()
                if float(value) < 1.0
            ]
        )
    if low_scores:
        low_scores.sort(key=lambda item: (item[1], item[0]))
        signature = ", ".join(f"{name}={value:.2f}" for name, value in low_scores[:3])
        return f"low scores: {signature}"
    if result.composite_quality_score is not None:
        return f"composite_quality_score={float(result.composite_quality_score):.2f} below threshold"
    return "score below threshold"


def _result_retrieval_warnings(result: CaseResult) -> list[str]:
    return sorted(
        {
            str(warning).strip()
            for diagnostic in result.retrieval_diagnostics
            for warning in diagnostic.warnings
            if str(warning).strip()
        }
    )


def _build_failure_reason(result: CaseResult) -> str:
    if result.runtime_errors:
        return ", ".join(result.runtime_errors)
    if result.response_errors:
        return ", ".join(result.response_errors)
    if result.judge_errors:
        return ", ".join(result.judge_errors)
    warnings = _result_retrieval_warnings(result)
    if warnings:
        return "retrieval_warning:" + ", ".join(warnings)
    if result.validator_reason:
        return f"validator:{result.validator_reason}"
    if result.llm_judge_reason:
        return _normalize_reason_text(result.llm_judge_reason)
    if result.gate_failures:
        return ", ".join(result.gate_failures)
    return _build_rule_score_signature(result)


def _tool_names_to_routes(tool_names: list[str]) -> list[str]:
    return _sort_routes({route_for_tool(tool_name) for tool_name in tool_names if route_for_tool(tool_name)})


def _observed_routes(result: CaseResult) -> list[str]:
    diagnostic_routes = _sort_routes(
        {
            str(item.route or route_for_tool(item.tool or "")).strip()
            for item in result.retrieval_diagnostics
            if str(item.route or route_for_tool(item.tool or "")).strip()
        }
    )
    if diagnostic_routes:
        return diagnostic_routes
    return _tool_names_to_routes(result.tool_calls)


def _build_category_pass_rates(results: list[CaseResult]) -> list[CategoryPassRate]:
    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    for result in results:
        per_category[result.category]["total"] += 1
        per_category[result.category]["passed"] += 1 if bool(result.release_pass) else 0

    rows: list[CategoryPassRate] = []
    for category in _sort_categories(list(per_category.keys())):
        counts = per_category[category]
        total_cases = counts["total"]
        pass_rate = (counts["passed"] / total_cases) if total_cases else 0.0
        rows.append(
            CategoryPassRate(
                category=category,
                passed_cases=counts["passed"],
                total_cases=total_cases,
                pass_rate=round(pass_rate, 4),
            )
        )
    return rows


def _build_planner_diagnostics_histogram(results: list[CaseResult]) -> list[PlannerDiagnosticsBucket]:
    counter: Counter[tuple[str, str, str | None, str | None]] = Counter()
    for result in results:
        if result.planner_diagnostics is None:
            counter[(result.category, "missing", "diagnostics_unavailable", None)] += 1
            continue
        counter[
            (
                result.category,
                str(result.planner_diagnostics.status or "missing"),
                result.planner_diagnostics.reason,
                result.planner_diagnostics.override_reason,
            )
        ] += 1

    rows = [
        PlannerDiagnosticsBucket(
            category=category,
            status=status,
            reason=reason,
            override_reason=override_reason,
            count=count,
        )
        for (category, status, reason, override_reason), count in counter.items()
    ]
    rows.sort(key=lambda item: (_category_sort_key(item.category), -item.count, item.status, item.reason or "", item.override_reason or ""))
    return rows


def _normalize_planner_error_code(error: str) -> str | None:
    normalized = str(error or "").strip().lower()
    if not normalized:
        return None
    if "structured_output_invocation_failed" in normalized or "structured output invocation failed" in normalized:
        return "structured_output_invocation_failed"
    if "sanitized_output_validation_failed" in normalized or "sanitized output validation failed" in normalized:
        return "sanitized_output_validation_failed"
    if "upload_route_dropped" in normalized or "dropped upload route because retriever is unavailable" in normalized:
        return "upload_route_dropped"
    if "output_validation_failed" in normalized or "output validation failed" in normalized:
        return "output_validation_failed"
    return None


def _build_planner_error_histogram(results: list[CaseResult]) -> list[PlannerErrorBucket]:
    counter: Counter[tuple[str, str]] = Counter()
    for result in results:
        for error in result.planner_errors:
            code = _normalize_planner_error_code(error)
            if code is not None:
                counter[(result.category, code)] += 1
    rows = [PlannerErrorBucket(category=category, error_code=error_code, count=count) for (category, error_code), count in counter.items()]
    rows.sort(key=lambda item: (_category_sort_key(item.category), _planner_error_sort_key(item.error_code), -item.count))
    return rows


def _build_retrieval_route_status_histogram(results: list[CaseResult]) -> list[RetrievalRouteStatusBucket]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for result in results:
        for diagnostic in result.retrieval_diagnostics:
            route = str(diagnostic.route or route_for_tool(diagnostic.tool or "")).strip()
            if not route:
                continue
            status = str(diagnostic.status or "unknown").strip() or "unknown"
            counter[(result.category, route, status)] += 1
    rows = [RetrievalRouteStatusBucket(category=category, route=route, status=status, count=count) for (category, route, status), count in counter.items()]
    rows.sort(key=lambda item: (_category_sort_key(item.category), _route_sort_key(item.route), -item.count, item.status))
    return rows


def _build_retrieval_warning_histogram(results: list[CaseResult]) -> list[RetrievalWarningBucket]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for result in results:
        for diagnostic in result.retrieval_diagnostics:
            route = str(diagnostic.route or route_for_tool(diagnostic.tool or "")).strip()
            if not route:
                continue
            for warning in diagnostic.warnings:
                normalized_warning = str(warning or "").strip()
                if normalized_warning:
                    counter[(result.category, route, normalized_warning)] += 1
    rows = [
        RetrievalWarningBucket(category=category, route=route, warning=warning, count=count)
        for (category, route, warning), count in counter.items()
    ]
    rows.sort(key=lambda item: (_category_sort_key(item.category), _route_sort_key(item.route), -item.count, item.warning))
    return rows


def _build_route_confusion(
    *,
    case_map: dict[str, BenchmarkCase],
    results: list[CaseResult],
) -> list[RouteConfusionBucket]:
    counter: Counter[
        tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    ] = Counter()
    for result in results:
        case = case_map.get(result.case_id)
        if case is None:
            continue
        expected_routes = set(_tool_names_to_routes(case.expected_tools))
        forbidden_routes = set(_tool_names_to_routes(case.forbidden_tools))
        observed_routes = set(_observed_routes(result))
        missing_expected = expected_routes.difference(observed_routes)
        forbidden_observed = observed_routes.intersection(forbidden_routes)
        unexpected_routes = observed_routes.difference(expected_routes).difference(forbidden_routes)
        if not (missing_expected or forbidden_observed or unexpected_routes):
            continue
        counter[
            (
                result.category,
                tuple(_sort_routes(expected_routes)),
                tuple(_sort_routes(observed_routes)),
                tuple(_sort_routes(missing_expected)),
                tuple(_sort_routes(unexpected_routes)),
                tuple(_sort_routes(forbidden_observed)),
            )
        ] += 1

    rows = [
        RouteConfusionBucket(
            category=category,
            expected_routes=list(expected_routes),
            observed_routes=list(observed_routes),
            missing_expected_routes=list(missing_expected_routes),
            unexpected_routes=list(unexpected_routes),
            forbidden_routes=list(forbidden_routes),
            count=count,
        )
        for (
            category,
            expected_routes,
            observed_routes,
            missing_expected_routes,
            unexpected_routes,
            forbidden_routes,
        ), count in counter.items()
    ]
    rows.sort(key=lambda item: (_category_sort_key(item.category), -item.count, ",".join(item.expected_routes), ",".join(item.observed_routes)))
    return rows


def _build_validator_reason_histogram(results: list[CaseResult]) -> list[ValidatorReasonBucket]:
    counter: Counter[tuple[str, str]] = Counter()
    totals_by_category: Counter[str] = Counter()
    for result in results:
        if result.release_pass:
            continue
        reason = str(result.validator_reason or "missing")
        counter[(result.category, reason)] += 1
        totals_by_category[result.category] += 1
    rows: list[ValidatorReasonBucket] = []
    for (category, reason), count in counter.items():
        total_failed = int(totals_by_category.get(category, 0))
        share = (count / total_failed) if total_failed else 0.0
        rows.append(ValidatorReasonBucket(category=category, reason=reason, count=count, share=round(share, 4)))
    rows.sort(key=lambda item: (_category_sort_key(item.category), -item.count, item.reason))
    return rows


def _build_synthesis_mode_histogram(results: list[CaseResult]) -> list[SynthesisModeBucket]:
    counter: Counter[tuple[str, str]] = Counter()
    for result in results:
        if result.synthesis_mode:
            counter[(result.category, result.synthesis_mode)] += 1
            continue
        breakdown = result.latency_breakdown
        if breakdown is None or not breakdown.synthesis_attempts:
            continue
        counter[(result.category, breakdown.synthesis_attempts[0].mode)] += 1
    rows = [SynthesisModeBucket(category=category, mode=mode, count=count) for (category, mode), count in counter.items()]
    rows.sort(key=lambda item: (_category_sort_key(item.category), -item.count, item.mode))
    return rows


def _extract_latency_stage_value(result: CaseResult, stage_name: str) -> int | None:
    breakdown = result.latency_breakdown
    if breakdown is None:
        return None
    if stage_name == "upload_retriever_build_ms":
        return breakdown.upload_retriever_build_ms
    return int(getattr(breakdown.stage_totals_ms, stage_name, 0) or 0)


def _build_stage_latency_analysis(
    results: list[CaseResult],
) -> tuple[list[StageLatencyPercentile], LatencyBreakdownCoverage]:
    available_results = [result for result in results if result.latency_breakdown is not None]
    total_cases = len(results)
    available_cases = len(available_results)
    coverage = LatencyBreakdownCoverage(
        available_cases=available_cases,
        total_cases=total_cases,
        coverage_rate=round((available_cases / total_cases), 4) if total_cases else 0.0,
    )
    rows: list[StageLatencyPercentile] = []
    for stage_name in _LATENCY_STAGE_FIELDS:
        values = [
            value
            for result in available_results
            for value in [_extract_latency_stage_value(result, stage_name)]
            if value is not None
        ]
        if not values:
            continue
        p50_value = _percentile(values, 0.50)
        p95_value = _percentile(values, 0.95)
        rows.append(
            StageLatencyPercentile(
                stage=stage_name,
                sample_count=len(values),
                p50_latency_ms=round(p50_value, 2) if p50_value is not None else None,
                p95_latency_ms=round(p95_value, 2) if p95_value is not None else None,
            )
        )
    return rows, coverage


def _build_analysis(*, case_map: dict[str, BenchmarkCase], results: list[CaseResult]) -> AnalysisStats:
    stage_latency_percentiles, latency_breakdown_coverage = _build_stage_latency_analysis(results)
    return AnalysisStats(
        category_pass_rates=_build_category_pass_rates(results),
        planner_diagnostics_histogram=_build_planner_diagnostics_histogram(results),
        planner_error_histogram=_build_planner_error_histogram(results),
        retrieval_route_status_histogram=_build_retrieval_route_status_histogram(results),
        retrieval_warning_histogram=_build_retrieval_warning_histogram(results),
        route_confusion=_build_route_confusion(case_map=case_map, results=results),
        validator_reason_histogram=_build_validator_reason_histogram(results),
        synthesis_mode_histogram=_build_synthesis_mode_histogram(results),
        stage_latency_percentiles=stage_latency_percentiles,
        latency_breakdown_coverage=latency_breakdown_coverage,
    )


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
    config: BenchmarkConfig,
    cases: list[BenchmarkCase],
    results: list[CaseResult],
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
    p50_latency = _percentile(latencies, 0.50)
    p95_latency = _percentile(latencies, 0.95)
    cost_values = [float(result.cost_usd) for result in results if result.cost_usd is not None]
    avg_cost = (sum(cost_values) / len(cost_values)) if cost_values else None
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
        {"case_id": result.case_id, "category": result.category, "reason": _build_failure_reason(result)}
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
    analysis = _build_analysis(case_map=case_map, results=results)
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
    ]
    overall_passed = all(gate.passed for gate in gates if gate.gate_type == "release")
    return RunSummary(
        run_id=run_id,
        endpoint=endpoint,
        fixtures_path=fixtures_path,
        config_path=config_path,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        mode="online",
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
            "cost_gate_eligible": metrics.cost_gate_eligible,
        },
    )


def _format_metric_value(value: float | int | None, *, decimals: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{decimals}f}"


def _render_analysis(lines: list[str], summary: RunSummary) -> None:
    lines.append("")
    lines.append("## Root Cause Breakdown")
    analysis = summary.analysis
    if analysis is None:
        lines.append("")
        lines.append("legacy run: unavailable")
        return

    lines.append("")
    lines.append("### Category Pass Rates")
    lines.append("")
    lines.append("| category | passed_cases | total_cases | pass_rate |")
    lines.append("|---|---:|---:|---:|")
    for row in analysis.category_pass_rates:
        lines.append(f"| {row.category} | {row.passed_cases} | {row.total_cases} | {row.pass_rate:.4f} |")

    lines.append("")
    lines.append("### Planner Diagnostics")
    lines.append("")
    lines.append("| category | status | reason | override_reason | count |")
    lines.append("|---|---|---|---|---:|")
    for row in analysis.planner_diagnostics_histogram:
        lines.append(f"| {row.category} | {row.status} | {row.reason or '-'} | {row.override_reason or '-'} | {row.count} |")

    lines.append("")
    lines.append("### Planner Errors")
    lines.append("")
    if analysis.planner_error_histogram:
        lines.append("| category | error_code | count |")
        lines.append("|---|---|---:|")
        for row in analysis.planner_error_histogram:
            lines.append(f"| {row.category} | {row.error_code} | {row.count} |")
    else:
        lines.append("No planner errors observed.")

    lines.append("")
    lines.append("### Retrieval Diagnostics")
    lines.append("")
    if analysis.retrieval_route_status_histogram:
        lines.append("| category | route | status | count |")
        lines.append("|---|---|---|---:|")
        for row in analysis.retrieval_route_status_histogram:
            lines.append(f"| {row.category} | {row.route} | {row.status} | {row.count} |")
    else:
        lines.append("No retrieval diagnostics observed.")

    lines.append("")
    lines.append("### Retrieval Warnings")
    lines.append("")
    if analysis.retrieval_warning_histogram:
        lines.append("| category | route | warning | count |")
        lines.append("|---|---|---|---:|")
        for row in analysis.retrieval_warning_histogram:
            lines.append(f"| {row.category} | {row.route} | {row.warning} | {row.count} |")
    else:
        lines.append("No retrieval warnings observed.")

    lines.append("")
    lines.append("### Route Confusion")
    lines.append("")
    if analysis.route_confusion:
        lines.append("| category | expected_routes | observed_routes | missing_expected_routes | unexpected_routes | forbidden_routes | count |")
        lines.append("|---|---|---|---|---|---|---:|")
        for row in analysis.route_confusion:
            lines.append(
                "| {category} | {expected} | {observed} | {missing} | {unexpected} | {forbidden} | {count} |".format(
                    category=row.category,
                    expected=", ".join(row.expected_routes) or "-",
                    observed=", ".join(row.observed_routes) or "-",
                    missing=", ".join(row.missing_expected_routes) or "-",
                    unexpected=", ".join(row.unexpected_routes) or "-",
                    forbidden=", ".join(row.forbidden_routes) or "-",
                    count=row.count,
                )
            )
    else:
        lines.append("No route confusion observed.")

    lines.append("")
    lines.append("### Validator Reasons")
    lines.append("")
    if analysis.validator_reason_histogram:
        lines.append("| category | reason | count | share |")
        lines.append("|---|---|---:|---:|")
        for row in analysis.validator_reason_histogram:
            lines.append(f"| {row.category} | {row.reason} | {row.count} | {row.share:.4f} |")
    else:
        lines.append("No failed cases for validator reason analysis.")

    lines.append("")
    lines.append("### Synthesis Modes")
    lines.append("")
    if analysis.synthesis_mode_histogram:
        lines.append("| category | mode | count |")
        lines.append("|---|---|---:|")
        for row in analysis.synthesis_mode_histogram:
            lines.append(f"| {row.category} | {row.mode} | {row.count} |")
    else:
        lines.append("No synthesis mode diagnostics observed.")

    lines.append("")
    lines.append("### Stage Latency")
    lines.append("")
    coverage = analysis.latency_breakdown_coverage
    if coverage is not None:
        lines.append(f"- Latency breakdown coverage: `{coverage.available_cases}/{coverage.total_cases}` cases (`{coverage.coverage_rate:.4f}`)")
    if not analysis.stage_latency_percentiles:
        lines.append("- unavailable for this run")
        return
    lines.append("")
    lines.append("| stage | sample_count | p50_latency_ms | p95_latency_ms |")
    lines.append("|---|---:|---:|---:|")
    for row in analysis.stage_latency_percentiles:
        lines.append(f"| {row.stage} | {row.sample_count} | {_format_metric_value(row.p50_latency_ms, decimals=2)} | {_format_metric_value(row.p95_latency_ms, decimals=2)} |")


def build_markdown_report(summary: RunSummary, results: list[CaseResult] | None = None) -> str:
    _ = results
    lines: list[str] = []
    lines.append(f"# Benchmark Report ({summary.run_id})")
    lines.append("")
    lines.append(f"- Mode: `{summary.mode}`")
    lines.append(f"- Endpoint: `{summary.endpoint}`")
    lines.append(f"- Fixtures: `{summary.fixtures_path}`")
    lines.append(f"- Release: `{'PASS' if summary.overall_passed else 'FAIL'}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key, value in (
        ("total_cases", summary.metrics.total_cases),
        ("scored_cases", summary.metrics.scored_cases),
        ("passed_cases", summary.metrics.passed_cases),
        ("product_passed_cases", summary.metrics.product_passed_cases),
        ("judge_passed_cases", summary.metrics.judge_passed_cases),
        ("release_passed_cases", summary.metrics.release_passed_cases),
        ("pass_rate", summary.metrics.pass_rate),
        ("product_pass_rate", summary.metrics.product_pass_rate),
        ("judge_pass_rate", summary.metrics.judge_pass_rate),
        ("release_pass_rate", summary.metrics.release_pass_rate),
        ("tool_precision", summary.metrics.tool_precision),
        ("tool_recall", summary.metrics.tool_recall),
        ("citation_compliance", summary.metrics.citation_compliance),
        ("p50_latency_ms", summary.metrics.p50_latency_ms),
        ("p95_latency_ms", summary.metrics.p95_latency_ms),
        ("avg_cost_per_case_usd", summary.metrics.avg_cost_per_case_usd),
        ("cost_gate_eligible", summary.metrics.cost_gate_eligible),
        ("llm_call_coverage_rate", summary.metrics.llm_call_coverage_rate),
        ("request_id_coverage_rate", summary.metrics.request_id_coverage_rate),
        ("judge_input_completeness_rate", summary.metrics.judge_input_completeness_rate),
        ("judge_min_score_failures", summary.metrics.judge_min_score_failures),
        ("deterministic_direct_usage_rate", summary.metrics.deterministic_direct_usage_rate),
        ("high_rule_low_judge_divergence_rate", summary.metrics.high_rule_low_judge_divergence_rate),
        ("planner_deterministic_rate", summary.metrics.planner_deterministic_rate),
        ("planner_llm_attempt_count", summary.metrics.planner_llm_attempt_count),
        ("planner_structured_success_rate", summary.metrics.planner_structured_success_rate),
        ("synthesis_structured_success_rate", summary.metrics.synthesis_structured_success_rate),
    ):
        lines.append(f"| {key} | {value} |")

    lines.append("")
    lines.append("## Gates")
    lines.append("")
    lines.append("| Gate | Type | Threshold | Actual | Passed | Status |")
    lines.append("|---|---|---:|---:|:---:|---|")
    for gate in summary.gates:
        lines.append(f"| {gate.name} | {gate.gate_type} | {gate.threshold} | {gate.actual} | {'Y' if gate.passed else 'N'} | {gate.status} |")

    _render_analysis(lines, summary)

    if summary.metrics.failures:
        lines.append("")
        lines.append("## Failures (Top 20)")
        lines.append("")
        lines.append("| case_id | category | reason |")
        lines.append("|---|---|---|")
        for failure in summary.metrics.failures[:20]:
            lines.append(f"| {failure['case_id']} | {failure['category']} | {failure['reason']} |")
    return "\n".join(lines) + "\n"


def write_run_outputs(*, output_dir: Path, results: list[CaseResult], summary: RunSummary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_dir / "raw_results.jsonl", results)
    (output_dir / "summary.json").write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(build_markdown_report(summary, results), encoding="utf-8")
    dump_jsonl(
        output_dir / "request_map.jsonl",
        [
            {
                "run_id": result.run_id,
                "case_id": result.case_id,
                "session_id": result.session_id,
                "request_id": result.request_id,
                "trace": result.trace,
                "created_at_utc": result.created_at_utc,
            }
            for result in results
        ],
    )
