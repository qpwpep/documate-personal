from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .scoring_rules import tool_confusion_counts
from .schemas import (
    BenchmarkCase,
    BenchmarkConfig,
    CaseResult,
    GateResult,
    RunSummary,
    SummaryStats,
    dump_jsonl,
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
        reason = ", ".join(result.errors) if result.errors else "score below threshold"
        failures.append(
            {
                "case_id": result.case_id,
                "category": result.category,
                "reason": reason,
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
        gates=gates,
        overall_passed=overall_passed,
        weights=config.weights.as_dict(),
        hard_gates=config.hard_gates.model_dump(),
        pricing=config.pricing.model_dump(),
        judge_enabled=config.judge_enabled,
        judge_model=config.judge_model,
    )


def build_markdown_report(summary: RunSummary) -> str:
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
    report_path.write_text(build_markdown_report(summary), encoding="utf-8")
