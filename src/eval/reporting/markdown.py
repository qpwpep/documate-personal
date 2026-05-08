from __future__ import annotations

from ..result_models import CaseResult
from ..summary_models import RunSummary


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
    lines.append("### Error Codes")
    lines.append("")
    if analysis.error_code_histogram:
        lines.append("| category | error_code | count |")
        lines.append("|---|---|---:|")
        for row in analysis.error_code_histogram:
            lines.append(f"| {row.category} | {row.error_code} | {row.count} |")
    else:
        lines.append("No standardized error codes observed.")

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
    lines.append("### Slack Delivery")
    lines.append("")
    if analysis.slack_delivery_status_histogram:
        lines.append("| category | status | count |")
        lines.append("|---|---|---:|")
        for row in analysis.slack_delivery_status_histogram:
            lines.append(f"| {row.category} | {row.status} | {row.count} |")
    else:
        lines.append("No live Slack delivery diagnostics observed.")

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
        ("hybrid_p95_latency_ms", summary.metrics.hybrid_p95_latency_ms),
        ("docs_only_p95_latency_ms", summary.metrics.docs_only_p95_latency_ms),
        ("avg_cost_per_case_usd", summary.metrics.avg_cost_per_case_usd),
        ("slack_delivery_required_cases", summary.metrics.slack_delivery_required_cases),
        ("slack_delivery_success_cases", summary.metrics.slack_delivery_success_cases),
        ("slack_delivery_success_rate", summary.metrics.slack_delivery_success_rate),
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
        ("planner_error_count", summary.metrics.planner_error_count),
        ("planner_error_case_count", summary.metrics.planner_error_case_count),
        ("planner_warning_count", summary.metrics.planner_warning_count),
        ("planner_duplicate_route_merge_count", summary.metrics.planner_duplicate_route_merge_count),
        ("planner_final_success_rate", summary.metrics.planner_final_success_rate),
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


__all__ = [
    "build_markdown_report",
]
