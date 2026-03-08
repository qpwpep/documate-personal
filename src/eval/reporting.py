from __future__ import annotations

from collections import Counter, defaultdict
import json
from datetime import datetime, timezone
from pathlib import Path

from .scoring_rules import tool_confusion_counts
from .schemas import (
    AnalysisStats,
    BenchmarkCase,
    BenchmarkConfig,
    CaseResult,
    CategoryPassRate,
    GateResult,
    LatencyBreakdownCoverage,
    PlannerDiagnosticsBucket,
    RetrievalRouteStatusBucket,
    RouteConfusionBucket,
    RunSummary,
    StageLatencyPercentile,
    SummaryStats,
    ValidatorReasonBucket,
    dump_jsonl,
)

_CATEGORY_ORDER: tuple[str, ...] = ("docs_only", "rag_only", "hybrid", "tool_action")
_ROUTE_ORDER: tuple[str, ...] = ("docs", "upload", "local")
_RETRIEVAL_TOOL_TO_ROUTE: dict[str, str] = {
    "tavily_search": "docs",
    "upload_search": "upload",
    "rag_search": "local",
}
_LATENCY_STAGE_FIELDS: tuple[str, ...] = (
    "upload_retriever_build_ms",
    "summarize_ms",
    "planner_ms",
    "retrieval_total_ms",
    "synthesis_total_ms",
    "validation_ms",
    "action_postprocess_ms",
)


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
    if value in _ROUTE_ORDER:
        return (_ROUTE_ORDER.index(value), value)
    return (len(_ROUTE_ORDER), value)


def _sort_categories(values: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=_category_sort_key)


def _sort_routes(values: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=_route_sort_key)


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
    if low_scores:
        low_scores.sort(key=lambda item: (item[1], item[0]))
        signature = ", ".join(f"{name}={value:.2f}" for name, value in low_scores[:3])
        return f"low scores: {signature}"
    if result.final_score is not None:
        return f"final_score={float(result.final_score):.2f} below threshold"
    return "score below threshold"


def _build_failure_reason(result: CaseResult) -> str:
    if result.runtime_errors:
        return ", ".join(result.runtime_errors)
    if result.response_errors:
        return ", ".join(result.response_errors)
    if result.judge_errors:
        return ", ".join(result.judge_errors)
    if result.validator_reason:
        return f"validator:{result.validator_reason}"
    if result.llm_judge_reason:
        return _normalize_reason_text(result.llm_judge_reason)
    return _build_rule_score_signature(result)


def _tool_names_to_routes(tool_names: list[str]) -> list[str]:
    return _sort_routes(
        {
            _RETRIEVAL_TOOL_TO_ROUTE[tool_name]
            for tool_name in tool_names
            if tool_name in _RETRIEVAL_TOOL_TO_ROUTE
        }
    )


def _observed_routes(result: CaseResult) -> list[str]:
    diagnostic_routes = _sort_routes(
        {
            str(item.route or _RETRIEVAL_TOOL_TO_ROUTE.get(item.tool, "")).strip()
            for item in result.retrieval_diagnostics
            if str(item.route or _RETRIEVAL_TOOL_TO_ROUTE.get(item.tool, "")).strip()
        }
    )
    if diagnostic_routes:
        return diagnostic_routes
    return _tool_names_to_routes(result.tool_calls)


def _build_category_pass_rates(results: list[CaseResult]) -> list[CategoryPassRate]:
    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    for result in results:
        per_category[result.category]["total"] += 1
        per_category[result.category]["passed"] += 1 if bool(result.passed) else 0

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
    rows.sort(
        key=lambda item: (
            _category_sort_key(item.category),
            -item.count,
            item.status,
            item.reason or "",
            item.override_reason or "",
        )
    )
    return rows


def _build_retrieval_route_status_histogram(results: list[CaseResult]) -> list[RetrievalRouteStatusBucket]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for result in results:
        for diagnostic in result.retrieval_diagnostics:
            route = str(diagnostic.route or _RETRIEVAL_TOOL_TO_ROUTE.get(diagnostic.tool, "")).strip()
            if not route:
                continue
            status = str(diagnostic.status or "unknown").strip() or "unknown"
            counter[(result.category, route, status)] += 1

    rows = [
        RetrievalRouteStatusBucket(
            category=category,
            route=route,
            status=status,
            count=count,
        )
        for (category, route, status), count in counter.items()
    ]
    rows.sort(
        key=lambda item: (
            _category_sort_key(item.category),
            _route_sort_key(item.route),
            -item.count,
            item.status,
        )
    )
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
    rows.sort(
        key=lambda item: (
            _category_sort_key(item.category),
            -item.count,
            ",".join(item.expected_routes),
            ",".join(item.observed_routes),
        )
    )
    return rows


def _build_validator_reason_histogram(results: list[CaseResult]) -> list[ValidatorReasonBucket]:
    counter: Counter[tuple[str, str]] = Counter()
    totals_by_category: Counter[str] = Counter()
    for result in results:
        if result.passed:
            continue
        reason = str(result.validator_reason or "missing")
        counter[(result.category, reason)] += 1
        totals_by_category[result.category] += 1

    rows: list[ValidatorReasonBucket] = []
    for (category, reason), count in counter.items():
        total_failed = int(totals_by_category.get(category, 0))
        share = (count / total_failed) if total_failed else 0.0
        rows.append(
            ValidatorReasonBucket(
                category=category,
                reason=reason,
                count=count,
                share=round(share, 4),
            )
        )
    rows.sort(key=lambda item: (_category_sort_key(item.category), -item.count, item.reason))
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


def _build_analysis(
    *,
    case_map: dict[str, BenchmarkCase],
    results: list[CaseResult],
) -> AnalysisStats:
    stage_latency_percentiles, latency_breakdown_coverage = _build_stage_latency_analysis(results)
    return AnalysisStats(
        category_pass_rates=_build_category_pass_rates(results),
        planner_diagnostics_histogram=_build_planner_diagnostics_histogram(results),
        retrieval_route_status_histogram=_build_retrieval_route_status_histogram(results),
        route_confusion=_build_route_confusion(case_map=case_map, results=results),
        validator_reason_histogram=_build_validator_reason_histogram(results),
        stage_latency_percentiles=stage_latency_percentiles,
        latency_breakdown_coverage=latency_breakdown_coverage,
    )


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

    scored_results = [result for result in results if result.final_score is not None]
    passed_results = [result for result in scored_results if bool(result.passed)]
    pass_rate = (len(passed_results) / len(scored_results)) if scored_results else 0.0

    tp_total = 0
    fp_total = 0
    fn_total = 0
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
        if not case:
            continue
        if case.require_official_citation or case.require_local_citation:
            citation_scores.append(float(result.rule_scores.get("citation_compliance", 0.0)))
    citation_compliance = (
        sum(citation_scores) / len(citation_scores) if citation_scores else 1.0
    )

    latencies = [int(result.latency_ms_e2e) for result in results if result.latency_ms_e2e is not None]
    p50_latency = _percentile(latencies, 0.50)
    p95_latency = _percentile(latencies, 0.95)

    cost_values = [float(result.cost_usd) for result in results if result.cost_usd is not None]
    avg_cost = (sum(cost_values) / len(cost_values)) if cost_values else None

    failures: list[dict[str, str]] = []
    for result in results:
        if result.passed:
            continue
        failures.append(
            {
                "case_id": result.case_id,
                "category": result.category,
                "reason": _build_failure_reason(result),
            }
        )

    metrics = SummaryStats(
        total_cases=len(results),
        scored_cases=len(scored_results),
        passed_cases=len(passed_results),
        pass_rate=round(pass_rate, 4),
        tool_precision=round(tool_precision, 4),
        tool_recall=round(tool_recall, 4),
        citation_compliance=round(citation_compliance, 4),
        p50_latency_ms=round(p50_latency, 2) if p50_latency is not None else None,
        p95_latency_ms=round(p95_latency, 2) if p95_latency is not None else None,
        avg_cost_per_case_usd=round(avg_cost, 8) if avg_cost is not None else None,
        failures=failures[:50],
    )
    analysis = _build_analysis(case_map=case_map, results=results)

    gates: list[GateResult] = []
    hard_gates = config.hard_gates

    gates.append(
        GateResult(
            name="pass_rate",
            threshold=hard_gates.pass_rate,
            actual=metrics.pass_rate,
            passed=metrics.pass_rate >= hard_gates.pass_rate,
        )
    )
    gates.append(
        GateResult(
            name="tool_precision",
            threshold=hard_gates.tool_precision,
            actual=metrics.tool_precision,
            passed=metrics.tool_precision >= hard_gates.tool_precision,
        )
    )
    gates.append(
        GateResult(
            name="tool_recall",
            threshold=hard_gates.tool_recall,
            actual=metrics.tool_recall,
            passed=metrics.tool_recall >= hard_gates.tool_recall,
        )
    )
    gates.append(
        GateResult(
            name="citation_compliance",
            threshold=hard_gates.citation_compliance,
            actual=metrics.citation_compliance,
            passed=metrics.citation_compliance >= hard_gates.citation_compliance,
        )
    )
    gates.append(
        GateResult(
            name="p95_latency_ms",
            threshold=hard_gates.p95_latency_ms,
            actual=metrics.p95_latency_ms,
            passed=metrics.p95_latency_ms is not None and metrics.p95_latency_ms <= hard_gates.p95_latency_ms,
        )
    )

    avg_cost_gate_passed = True
    if metrics.avg_cost_per_case_usd is not None:
        avg_cost_gate_passed = metrics.avg_cost_per_case_usd <= hard_gates.avg_cost_per_case_usd
    gates.append(
        GateResult(
            name="avg_cost_per_case_usd",
            threshold=hard_gates.avg_cost_per_case_usd,
            actual=metrics.avg_cost_per_case_usd,
            passed=avg_cost_gate_passed,
        )
    )

    overall_passed = all(gate.passed for gate in gates)

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
        lines.append(
            f"| {row.category} | {row.passed_cases} | {row.total_cases} | {row.pass_rate:.4f} |"
        )

    lines.append("")
    lines.append("### Planner Diagnostics")
    lines.append("")
    lines.append("| category | status | reason | override_reason | count |")
    lines.append("|---|---|---|---|---:|")
    for row in analysis.planner_diagnostics_histogram:
        lines.append(
            f"| {row.category} | {row.status} | {row.reason or '-'} | "
            f"{row.override_reason or '-'} | {row.count} |"
        )

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
    lines.append("### Route Confusion")
    lines.append("")
    if analysis.route_confusion:
        lines.append(
            "| category | expected_routes | observed_routes | missing_expected_routes | "
            "unexpected_routes | forbidden_routes | count |"
        )
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
    lines.append("### Stage Latency")
    lines.append("")
    coverage = analysis.latency_breakdown_coverage
    if coverage is not None:
        lines.append(
            "- Latency breakdown coverage: "
            f"`{coverage.available_cases}/{coverage.total_cases}` cases "
            f"(`{coverage.coverage_rate:.4f}`)"
        )
    if not analysis.stage_latency_percentiles:
        lines.append("- unavailable for this run")
        return

    lines.append("")
    lines.append("| stage | sample_count | p50_latency_ms | p95_latency_ms |")
    lines.append("|---|---:|---:|---:|")
    for row in analysis.stage_latency_percentiles:
        lines.append(
            f"| {row.stage} | {row.sample_count} | "
            f"{_format_metric_value(row.p50_latency_ms, decimals=2)} | "
            f"{_format_metric_value(row.p95_latency_ms, decimals=2)} |"
        )


def build_markdown_report(summary: RunSummary, results: list[CaseResult] | None = None) -> str:
    _ = results
    lines: list[str] = []
    lines.append(f"# Benchmark Report ({summary.run_id})")
    lines.append("")
    lines.append(f"- Mode: `{summary.mode}`")
    lines.append(f"- Endpoint: `{summary.endpoint}`")
    lines.append(f"- Fixtures: `{summary.fixtures_path}`")
    lines.append(f"- Overall: `{'PASS' if summary.overall_passed else 'FAIL'}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| total_cases | {summary.metrics.total_cases} |")
    lines.append(f"| scored_cases | {summary.metrics.scored_cases} |")
    lines.append(f"| passed_cases | {summary.metrics.passed_cases} |")
    lines.append(f"| pass_rate | {summary.metrics.pass_rate:.4f} |")
    lines.append(f"| tool_precision | {summary.metrics.tool_precision:.4f} |")
    lines.append(f"| tool_recall | {summary.metrics.tool_recall:.4f} |")
    lines.append(f"| citation_compliance | {summary.metrics.citation_compliance:.4f} |")
    lines.append(f"| p50_latency_ms | {summary.metrics.p50_latency_ms} |")
    lines.append(f"| p95_latency_ms | {summary.metrics.p95_latency_ms} |")
    lines.append(f"| avg_cost_per_case_usd | {summary.metrics.avg_cost_per_case_usd} |")
    lines.append("")
    lines.append("## Hard Gates")
    lines.append("")
    lines.append("| Gate | Threshold | Actual | Passed |")
    lines.append("|---|---:|---:|:---:|")
    for gate in summary.gates:
        lines.append(
            f"| {gate.name} | {gate.threshold} | {gate.actual} | {'Y' if gate.passed else 'N'} |"
        )

    _render_analysis(lines, summary)

    if summary.metrics.failures:
        lines.append("")
        lines.append("## Failures (Top 20)")
        lines.append("")
        lines.append("| case_id | category | reason |")
        lines.append("|---|---|---|")
        for failure in summary.metrics.failures[:20]:
            lines.append(
                f"| {failure['case_id']} | {failure['category']} | {failure['reason']} |"
            )
    return "\n".join(lines) + "\n"


def write_run_outputs(
    *,
    output_dir: Path,
    results: list[CaseResult],
    summary: RunSummary,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_results_path = output_dir / "raw_results.jsonl"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    dump_jsonl(raw_results_path, results)
    summary_path.write_text(
        json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(build_markdown_report(summary, results), encoding="utf-8")
