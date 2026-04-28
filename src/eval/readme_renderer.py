from __future__ import annotations

import os
from pathlib import Path

from .history_loader import StoredRun, latest_run_pointer_name, suite_label
from .summary_models import RunSummary, RunTrack


README_HISTORY_START = "## 9. 최신 벤치마크 결과"
README_HISTORY_END = "## 11. 테스트 및 검증"
HISTORY_TABLE_METRICS = [
    "pass_rate",
    "tool_precision",
    "tool_recall",
    "citation_compliance",
    "p50_latency_ms",
    "p95_latency_ms",
    "avg_cost_per_case_usd",
]


def format_metric_value(metric_key: str, value: float | int | None) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if metric_key in {"pass_rate", "tool_precision", "tool_recall", "citation_compliance"}:
        return f"{numeric:.4f}"
    if metric_key in {"p50_latency_ms", "p95_latency_ms"}:
        return f"{numeric:.1f}"
    if metric_key == "avg_cost_per_case_usd":
        return f"{numeric:.8f}"
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def format_gate_threshold(metric_key: str, value: float | int) -> str:
    numeric = float(value)
    if metric_key in {"pass_rate", "tool_precision", "tool_recall", "citation_compliance"}:
        return f"{numeric:.2f}"
    if metric_key == "avg_cost_per_case_usd":
        return f"{numeric:.3f}"
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def format_delta(metric_key: str, value: float) -> str:
    if metric_key in {"pass_rate", "tool_precision", "tool_recall", "citation_compliance"}:
        return f"{value:+.4f}"
    if metric_key in {"p50_latency_ms", "p95_latency_ms"}:
        return f"{value:+.1f}"
    if metric_key == "avg_cost_per_case_usd":
        return f"{value:+.8f}"
    return f"{value:+.4f}"


def build_history_readme_block(
    *,
    track: RunTrack,
    latest: StoredRun,
    comparable_runs: list[StoredRun],
    readme_path: Path,
    output_root: Path,
    svg_path: Path,
) -> str:
    current_suite_label = suite_label(latest.summary.fixtures_path)
    previous = comparable_runs[-2] if len(comparable_runs) > 1 else None
    passed_gates, failed_gates = _gate_lists(latest.summary)
    svg_markdown_path = _relative_markdown_path(svg_path, readme_path.parent)
    pointer_name = latest_run_pointer_name(track)

    lines: list[str] = []
    lines.append("## 9. 최신 벤치마크 결과")
    lines.append("")
    lines.append(f"기준 런은 `output/benchmarks/{pointer_name}`가 가리키는 `{latest.run_id}`입니다.")
    lines.append("")
    lines.append(f"- run_id: `{latest.run_id}`")
    lines.append(f"- track: `{track}`")
    lines.append(f"- generated_at_utc: `{latest.summary.generated_at_utc}`")
    lines.append(f"- endpoint: `{latest.summary.endpoint}`")
    lines.append(f"- fixtures: `{latest.summary.fixtures_path}`")
    lines.append(f"- judge_model: `{latest.summary.judge_model}`")
    lines.append(f"- overall: `{'PASS' if latest.summary.overall_passed else 'FAIL'}`")
    lines.append("")
    lines.append("### 9.1 Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| total_cases | {latest.metrics.total_cases} |")
    lines.append(f"| scored_cases | {latest.metrics.scored_cases} |")
    lines.append(f"| passed_cases | {latest.metrics.passed_cases} |")
    for metric_key in HISTORY_TABLE_METRICS:
        lines.append(
            f"| {metric_key} | {format_metric_value(metric_key, getattr(latest.metrics, metric_key))} |"
        )
    lines.append("")
    lines.append("### 9.2 Hard Gates")
    lines.append("")
    lines.append("| Gate | Threshold | Actual | Passed |")
    lines.append("|---|---:|---:|:---:|")
    for gate in latest.summary.gates:
        lines.append(
            "| {name} | {threshold} | {actual} | {passed} |".format(
                name=gate.name,
                threshold=format_gate_threshold(gate.name, gate.threshold),
                actual=format_metric_value(gate.name, gate.actual),
                passed="Y" if gate.passed else "N",
            )
        )
    lines.append("")
    lines.append(
        "최신 런은 {passed} Hard Gate를 통과했지만 {failed}는 아직 기준에 못 미칩니다. "
        "개별 리포트는 로컬 `output/benchmarks/` 또는 release artifact에서 확인합니다.".format(
            passed=_quoted_metric_names(passed_gates) if passed_gates else "아직 어떤",
            failed=_quoted_metric_names(failed_gates) if failed_gates else "추가 실패 항목",
        )
    )
    lines.append("")
    lines.append("## 10. 최근 벤치마크 이력 및 추세")
    lines.append("")
    if previous is None:
        lines.append(
            f"저장소에 남아 있는 비교 가능한 {current_suite_label} 런은 현재 `{latest.run_id}` 하나뿐입니다."
        )
    else:
        lines.append(
            "저장소에 남아 있는 {count}개 {suite} 런 기준으로 보면, 최신 `{latest_run}` 런은 "
            "직전 `{previous_run}` 대비 `pass_rate` {pass_rate}, `tool_precision` {tool_precision}, "
            "`tool_recall` {tool_recall}, `citation_compliance` {citation}, `p95_latency_ms` {p95_latency}, "
            "`avg_cost_per_case_usd` {avg_cost} 변화를 보였습니다. overall 상태는 여전히 `{overall}`입니다.".format(
                count=len(comparable_runs),
                suite=current_suite_label,
                latest_run=latest.run_id,
                previous_run=previous.run_id,
                pass_rate=format_delta(
                    "pass_rate",
                    float(latest.metrics.pass_rate) - float(previous.metrics.pass_rate),
                ),
                tool_precision=format_delta(
                    "tool_precision",
                    float(latest.metrics.tool_precision) - float(previous.metrics.tool_precision),
                ),
                tool_recall=format_delta(
                    "tool_recall",
                    float(latest.metrics.tool_recall) - float(previous.metrics.tool_recall),
                ),
                citation=format_delta(
                    "citation_compliance",
                    float(latest.metrics.citation_compliance) - float(previous.metrics.citation_compliance),
                ),
                p95_latency=format_delta(
                    "p95_latency_ms",
                    float(latest.metrics.p95_latency_ms or 0.0)
                    - float(previous.metrics.p95_latency_ms or 0.0),
                ),
                avg_cost=format_delta(
                    "avg_cost_per_case_usd",
                    float(latest.metrics.avg_cost_per_case_usd or 0.0)
                    - float(previous.metrics.avg_cost_per_case_usd or 0.0),
                ),
                overall="PASS" if latest.summary.overall_passed else "FAIL",
            )
        )
    lines.append("")
    lines.append(
        "| run_id | generated_at_utc | overall | pass_rate | tool_precision | tool_recall | "
        "citation_compliance | p50_latency_ms | p95_latency_ms | avg_cost_per_case_usd | 변화 |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for index, run in enumerate(comparable_runs):
        previous_run = comparable_runs[index - 1] if index > 0 else None
        lines.append(
            "| `{run_id}` | `{generated_at}` | `{overall}` | {pass_rate} | {tool_precision} | {tool_recall} | "
            "{citation} | {p50} | {p95} | {avg_cost} | {delta} |".format(
                run_id=run.run_id,
                generated_at=run.summary.generated_at_utc,
                overall="PASS" if run.summary.overall_passed else "FAIL",
                pass_rate=format_metric_value("pass_rate", run.metrics.pass_rate),
                tool_precision=format_metric_value("tool_precision", run.metrics.tool_precision),
                tool_recall=format_metric_value("tool_recall", run.metrics.tool_recall),
                citation=format_metric_value("citation_compliance", run.metrics.citation_compliance),
                p50=format_metric_value("p50_latency_ms", run.metrics.p50_latency_ms),
                p95=format_metric_value("p95_latency_ms", run.metrics.p95_latency_ms),
                avg_cost=format_metric_value("avg_cost_per_case_usd", run.metrics.avg_cost_per_case_usd),
                delta=_build_delta_text(run, previous_run),
            )
        )
    lines.append("")
    lines.append(f"![DocuMate benchmark history]({svg_markdown_path})")
    lines.append("")

    report_links = [
        f"[run {run.run_id}]({_relative_markdown_path(output_root / run.run_id / 'report.md', readme_path.parent)})"
        for run in comparable_runs
    ]
    lines.append(
        f"저장소에 남아 있는 {len(comparable_runs)}개 {current_suite_label} 런 기준 trend chart입니다. "
        f"상세 수치는 {', '.join(report_links)}에서 다시 확인할 수 있습니다."
    )
    return "\n".join(lines).rstrip() + "\n"


def replace_history_block(readme_text: str, history_block: str) -> str:
    start_index = readme_text.find(README_HISTORY_START)
    end_index = readme_text.find(README_HISTORY_END)
    if start_index < 0 or end_index < 0 or end_index <= start_index:
        raise ValueError("README benchmark history section markers were not found.")

    prefix = readme_text[:start_index].rstrip()
    suffix = readme_text[end_index:].lstrip("\n")
    return prefix + "\n\n" + history_block.rstrip() + "\n\n" + suffix


def _relative_markdown_path(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path, base_dir)).as_posix()


def _build_delta_text(current: StoredRun, previous: StoredRun | None) -> str:
    if previous is None:
        return "기준 런"

    parts: list[str] = []
    for metric_key in HISTORY_TABLE_METRICS:
        current_value = getattr(current.metrics, metric_key)
        previous_value = getattr(previous.metrics, metric_key)
        if current_value is None or previous_value is None:
            continue
        parts.append(
            f"{metric_key} {format_delta(metric_key, float(current_value) - float(previous_value))}"
        )
    return "`" + "; ".join(parts) + "`"


def _gate_lists(summary: RunSummary) -> tuple[list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []
    for gate in summary.gates:
        if gate.passed:
            passed.append(gate.name)
        else:
            failed.append(gate.name)
    return passed, failed


def _quoted_metric_names(metric_names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in metric_names)
