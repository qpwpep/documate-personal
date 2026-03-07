from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from xml.sax.saxutils import escape

from .schemas import RunSummary


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

SVG_COLORS = [
    ("#b07a4f", "#8c5d36"),
    ("#6b7280", "#4b5563"),
    ("#127475", "#0f5c5d"),
    ("#a23b72", "#7f2858"),
    ("#2f6fed", "#1f4db5"),
    ("#2f855a", "#276749"),
    ("#d97706", "#b45309"),
    ("#a16207", "#854d0e"),
]


@dataclass(frozen=True)
class StoredRun:
    summary: RunSummary
    generated_at: datetime

    @property
    def run_id(self) -> str:
        return self.summary.run_id

    @property
    def metrics(self):
        return self.summary.metrics


@dataclass(frozen=True)
class MetricSpec:
    key: str
    direction: str
    higher_is_better: bool
    gate_key: str | None = None


SVG_METRICS = [
    MetricSpec("pass_rate", "higher is better", True, "pass_rate"),
    MetricSpec("tool_precision", "higher is better", True, "tool_precision"),
    MetricSpec("tool_recall", "higher is better", True, "tool_recall"),
    MetricSpec("citation_compliance", "higher is better", True, "citation_compliance"),
    MetricSpec("p95_latency_ms", "lower is better", False, "p95_latency_ms"),
    MetricSpec("avg_cost_per_case_usd", "lower is better", False, "avg_cost_per_case_usd"),
]


def _parse_generated_at(summary: RunSummary) -> datetime:
    return datetime.fromisoformat(summary.generated_at_utc)


def _read_summary(path: Path) -> RunSummary:
    return RunSummary(**json.loads(path.read_text(encoding="utf-8")))


def load_history_runs(output_root: Path) -> list[StoredRun]:
    runs: list[StoredRun] = []
    for entry in output_root.iterdir():
        if not entry.is_dir():
            continue
        summary_path = entry / "summary.json"
        if not summary_path.exists():
            continue
        summary = _read_summary(summary_path)
        runs.append(StoredRun(summary=summary, generated_at=_parse_generated_at(summary)))
    runs.sort(key=lambda item: item.generated_at)
    return runs


def load_latest_run_id(output_root: Path) -> str | None:
    latest_path = output_root / "latest_run.txt"
    if not latest_path.exists():
        return None
    latest_run_id = latest_path.read_text(encoding="utf-8").strip()
    return latest_run_id or None


def select_comparable_runs(
    runs: list[StoredRun],
    *,
    latest_run_id: str | None = None,
) -> tuple[StoredRun, list[StoredRun]]:
    if not runs:
        raise ValueError("No benchmark summaries were found.")

    latest = next((run for run in runs if run.run_id == latest_run_id), runs[-1])
    fixtures_path = latest.summary.fixtures_path
    total_cases = latest.metrics.total_cases

    comparable = [
        run
        for run in runs
        if run.summary.fixtures_path == fixtures_path and run.metrics.total_cases == total_cases
    ]
    comparable.sort(key=lambda item: item.generated_at)
    return latest, comparable


def _suite_label(fixtures_path: str) -> str:
    file_name = Path(fixtures_path.replace("\\", "/")).name
    if file_name == "cases.generated.jsonl":
        return "generated-suite"
    return file_name


def _relative_markdown_path(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path, base_dir)).as_posix()


def _format_metric_value(metric_key: str, value: float | int | None) -> str:
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


def _format_gate_threshold(metric_key: str, value: float | int) -> str:
    numeric = float(value)
    if metric_key in {"pass_rate", "tool_precision", "tool_recall", "citation_compliance"}:
        return f"{numeric:.2f}"
    if metric_key == "avg_cost_per_case_usd":
        return f"{numeric:.3f}"
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def _format_delta(metric_key: str, value: float) -> str:
    if metric_key in {"pass_rate", "tool_precision", "tool_recall", "citation_compliance"}:
        return f"{value:+.4f}"
    if metric_key in {"p50_latency_ms", "p95_latency_ms"}:
        return f"{value:+.1f}"
    if metric_key == "avg_cost_per_case_usd":
        return f"{value:+.8f}"
    return f"{value:+.4f}"


def _build_delta_text(current: StoredRun, previous: StoredRun | None) -> str:
    if previous is None:
        return "기준 런"

    parts: list[str] = []
    for metric_key in HISTORY_TABLE_METRICS:
        current_value = getattr(current.metrics, metric_key)
        previous_value = getattr(previous.metrics, metric_key)
        if current_value is None or previous_value is None:
            continue
        parts.append(f"{metric_key} {_format_delta(metric_key, float(current_value) - float(previous_value))}")
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


def build_history_readme_block(
    *,
    latest: StoredRun,
    comparable_runs: list[StoredRun],
    readme_path: Path,
    output_root: Path,
    svg_path: Path,
) -> str:
    suite_label = _suite_label(latest.summary.fixtures_path)
    previous = comparable_runs[-2] if len(comparable_runs) > 1 else None
    passed_gates, failed_gates = _gate_lists(latest.summary)
    latest_report_path = _relative_markdown_path(output_root / latest.run_id / "report.md", readme_path.parent)
    svg_markdown_path = _relative_markdown_path(svg_path, readme_path.parent)

    lines: list[str] = []
    lines.append("## 9. 최신 벤치마크 결과")
    lines.append("")
    lines.append(f"기준 런은 `output/benchmarks/latest_run.txt`가 가리키는 `{latest.run_id}`입니다.")
    lines.append("")
    lines.append(f"- run_id: `{latest.run_id}`")
    lines.append(f"- generated_at_utc: `{latest.summary.generated_at_utc}`")
    lines.append(f"- endpoint: `{latest.summary.endpoint}`")
    lines.append(f"- fixtures: `{latest.summary.fixtures_path}`")
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
        lines.append(f"| {metric_key} | {_format_metric_value(metric_key, getattr(latest.metrics, metric_key))} |")
    lines.append("")
    lines.append("### 9.2 Hard Gates")
    lines.append("")
    lines.append("| Gate | Threshold | Actual | Passed |")
    lines.append("|---|---:|---:|:---:|")
    for gate in latest.summary.gates:
        lines.append(
            "| {name} | {threshold} | {actual} | {passed} |".format(
                name=gate.name,
                threshold=_format_gate_threshold(gate.name, gate.threshold),
                actual=_format_metric_value(gate.name, gate.actual),
                passed="Y" if gate.passed else "N",
            )
        )
    lines.append("")
    lines.append(
        "최신 런은 {passed} Hard Gate를 통과했지만 {failed}는 아직 기준에 못 미칩니다. "
        "상세 목록은 [latest report]({report_path})를 참고하세요.".format(
            passed=_quoted_metric_names(passed_gates) if passed_gates else "아직 어떤",
            failed=_quoted_metric_names(failed_gates) if failed_gates else "추가 실패 항목",
            report_path=latest_report_path,
        )
    )
    lines.append("")
    lines.append("## 10. 최근 벤치마크 이력 및 추세")
    lines.append("")
    if previous is None:
        lines.append(
            f"저장소에 남아 있는 비교 가능한 {suite_label} 런은 현재 `{latest.run_id}` 하나뿐입니다."
        )
    else:
        lines.append(
            "저장소에 남아 있는 {count}개 {suite} 런 기준으로 보면, 최신 `{latest_run}` 런은 "
            "직전 `{previous_run}` 대비 `pass_rate` {pass_rate}, `tool_precision` {tool_precision}, "
            "`tool_recall` {tool_recall}, `citation_compliance` {citation}, `p95_latency_ms` {p95_latency}, "
            "`avg_cost_per_case_usd` {avg_cost} 변화를 보였습니다. overall 상태는 여전히 `{overall}`입니다.".format(
                count=len(comparable_runs),
                suite=suite_label,
                latest_run=latest.run_id,
                previous_run=previous.run_id,
                pass_rate=_format_delta(
                    "pass_rate",
                    float(latest.metrics.pass_rate) - float(previous.metrics.pass_rate),
                ),
                tool_precision=_format_delta(
                    "tool_precision",
                    float(latest.metrics.tool_precision) - float(previous.metrics.tool_precision),
                ),
                tool_recall=_format_delta(
                    "tool_recall",
                    float(latest.metrics.tool_recall) - float(previous.metrics.tool_recall),
                ),
                citation=_format_delta(
                    "citation_compliance",
                    float(latest.metrics.citation_compliance) - float(previous.metrics.citation_compliance),
                ),
                p95_latency=_format_delta(
                    "p95_latency_ms",
                    float(latest.metrics.p95_latency_ms or 0.0) - float(previous.metrics.p95_latency_ms or 0.0),
                ),
                avg_cost=_format_delta(
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
                pass_rate=_format_metric_value("pass_rate", run.metrics.pass_rate),
                tool_precision=_format_metric_value("tool_precision", run.metrics.tool_precision),
                tool_recall=_format_metric_value("tool_recall", run.metrics.tool_recall),
                citation=_format_metric_value("citation_compliance", run.metrics.citation_compliance),
                p50=_format_metric_value("p50_latency_ms", run.metrics.p50_latency_ms),
                p95=_format_metric_value("p95_latency_ms", run.metrics.p95_latency_ms),
                avg_cost=_format_metric_value("avg_cost_per_case_usd", run.metrics.avg_cost_per_case_usd),
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
        f"저장소에 남아 있는 {len(comparable_runs)}개 {suite_label} 런 기준 trend chart입니다. "
        f"상세 수치는 {', '.join(report_links)}에서 다시 확인할 수 있습니다."
    )
    return "\n".join(lines).rstrip() + "\n"


def _replace_history_block(readme_text: str, history_block: str) -> str:
    start_index = readme_text.find(README_HISTORY_START)
    end_index = readme_text.find(README_HISTORY_END)
    if start_index < 0 or end_index < 0 or end_index <= start_index:
        raise ValueError("README benchmark history section markers were not found.")

    prefix = readme_text[:start_index].rstrip()
    suffix = readme_text[end_index:].lstrip("\n")
    return prefix + "\n\n" + history_block.rstrip() + "\n\n" + suffix


def _round_up(value: float, step: float) -> float:
    if step <= 0:
        return value
    return ceil(value / step) * step


def _scale_max(metric_key: str, values: list[float], gate_value: float | None) -> float:
    max_value = max(values) if values else 1.0
    if metric_key in {"pass_rate", "tool_precision", "tool_recall", "citation_compliance"}:
        return 1.0
    if metric_key == "p95_latency_ms":
        baseline = max(max_value, gate_value or 0.0) * 1.1
        return max(1000.0, _round_up(baseline, 5000.0))
    if metric_key == "avg_cost_per_case_usd":
        baseline = max_value * 1.15
        return max(0.0010, _round_up(baseline, 0.0002))
    return max_value or 1.0


def _svg_axis_label(metric_key: str, value: float) -> str:
    if metric_key in {"pass_rate", "tool_precision", "tool_recall", "citation_compliance"}:
        return f"{value:.1f}"
    if metric_key == "p95_latency_ms":
        return str(int(value))
    if metric_key == "avg_cost_per_case_usd":
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _map_y(value: float, minimum: float, maximum: float, top: float, bottom: float) -> float:
    if maximum <= minimum:
        return bottom
    ratio = (value - minimum) / (maximum - minimum)
    return bottom - ratio * (bottom - top)


def _svg_value_label(metric_key: str, value: float) -> str:
    return _format_metric_value(metric_key, value)


def build_history_svg(comparable_runs: list[StoredRun]) -> str:
    if not comparable_runs:
        raise ValueError("No comparable runs were supplied for SVG generation.")

    width = 1200
    legend_columns = min(3, len(comparable_runs))
    legend_rows = ceil(len(comparable_runs) / legend_columns)
    legend_start_y = 118
    legend_row_height = 46
    panel_top = legend_start_y + legend_rows * legend_row_height + 26
    height = panel_top + 3 * 270

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
    )
    lines.append("  <title id=\"title\">DocuMate benchmark history</title>")
    lines.append(
        "  <desc id=\"desc\">Stored benchmark runs "
        + ", ".join(run.run_id for run in comparable_runs)
        + " across pass rate, tool precision, tool recall, citation compliance, p95 latency, and average cost.</desc>"
    )
    lines.append("  <defs>")
    lines.append("    <style>")
    lines.append("      .bg { fill: #f5f1e8; }")
    lines.append("      .panel { fill: #fffdfa; stroke: #d9cfbf; stroke-width: 1; }")
    lines.append("      .title { font: 700 28px 'Segoe UI', 'Noto Sans KR', sans-serif; fill: #1f2933; }")
    lines.append("      .subtitle { font: 400 15px 'Segoe UI', 'Noto Sans KR', sans-serif; fill: #52606d; }")
    lines.append("      .metric { font: 700 19px 'Segoe UI', 'Noto Sans KR', sans-serif; fill: #243b53; }")
    lines.append("      .axis { font: 400 12px 'Segoe UI', 'Noto Sans KR', sans-serif; fill: #52606d; }")
    lines.append("      .tick { font: 600 11px 'Segoe UI', 'Noto Sans KR', sans-serif; fill: #334e68; }")
    lines.append("      .value { font: 700 11px 'Segoe UI', 'Noto Sans KR', sans-serif; fill: #102a43; }")
    lines.append("      .legend { font: 600 12px 'Segoe UI', 'Noto Sans KR', sans-serif; fill: #334e68; }")
    lines.append("      .legend-sub { font: 400 11px 'Segoe UI', 'Noto Sans KR', sans-serif; fill: #52606d; }")
    lines.append("      .guide { stroke: #d9cfbf; stroke-width: 1; }")
    lines.append("      .trend { stroke: #486581; fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }")
    lines.append("      .gate { stroke: #7c2d12; stroke-width: 2; stroke-dasharray: 8 6; }")
    lines.append("      .gate-label { font: 600 11px 'Segoe UI', 'Noto Sans KR', sans-serif; fill: #7c2d12; }")
    lines.append("    </style>")
    lines.append("  </defs>")
    lines.append("")
    lines.append(f'  <rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="24"/>')
    lines.append('  <text class="title" x="60" y="58">DocuMate Benchmark Trend</text>')
    lines.append(
        '  <text class="subtitle" x="60" y="86">'
        + escape(f"{len(comparable_runs)} comparable {_suite_label(comparable_runs[-1].summary.fixtures_path)} runs, 6 key metrics.")
        + "</text>"
    )

    legend_cell_width = 360
    for index, run in enumerate(comparable_runs):
        fill, stroke = SVG_COLORS[index % len(SVG_COLORS)]
        column = index % legend_columns
        row = index // legend_columns
        origin_x = 60 + column * legend_cell_width
        origin_y = legend_start_y + row * legend_row_height
        lines.append(
            f'  <circle cx="{origin_x}" cy="{origin_y}" r="7" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
        lines.append(f'  <text class="legend" x="{origin_x + 16}" y="{origin_y + 4}">{escape(run.run_id)}</text>')
        lines.append(
            f'  <text class="legend-sub" x="{origin_x + 16}" y="{origin_y + 21}">'
            f"{run.generated_at.strftime('%m-%d %H:%M')}, {'PASS' if run.summary.overall_passed else 'FAIL'}</text>"
        )

    panel_positions = [
        (40, panel_top),
        (620, panel_top),
        (40, panel_top + 270),
        (620, panel_top + 270),
        (40, panel_top + 540),
        (620, panel_top + 540),
    ]

    for spec, (panel_x, panel_y) in zip(SVG_METRICS, panel_positions):
        plot_left = panel_x + 90
        plot_right = panel_x + 460
        plot_top = panel_y + 46
        plot_bottom = panel_y + 184
        gate_value = None
        if spec.gate_key:
            gate_value = float(comparable_runs[-1].summary.hard_gates.get(spec.gate_key, 0.0))
        values = [float(getattr(run.metrics, spec.key) or 0.0) for run in comparable_runs]
        scale_min = 0.0
        scale_max = _scale_max(spec.key, values, gate_value)
        x_padding = 20.0
        usable_width = (plot_right - plot_left) - 2 * x_padding
        if len(comparable_runs) == 1:
            xs = [plot_left + (plot_right - plot_left) / 2]
        else:
            xs = [plot_left + x_padding + (usable_width * index / (len(comparable_runs) - 1)) for index in range(len(comparable_runs))]
        ys = [_map_y(value, scale_min, scale_max, plot_top, plot_bottom) for value in values]

        lines.append(f'  <g transform="translate(0,0)">')
        lines.append(f'    <rect class="panel" x="{panel_x}" y="{panel_y}" width="540" height="230" rx="18"/>')
        lines.append(f'    <text class="metric" x="{panel_x + 26}" y="{panel_y + 34}">{escape(spec.key)}</text>')

        subtitle = spec.direction
        if spec.key == "avg_cost_per_case_usd" and gate_value is not None and gate_value > scale_max:
            subtitle = f"{subtitle}; gate {_format_gate_threshold(spec.key, gate_value)} is off chart"
        lines.append(f'    <text class="axis" x="{panel_x + 26}" y="{panel_y + 56}">{escape(subtitle)}</text>')

        lines.append(f'    <line class="guide" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>')
        lines.append(f'    <line class="guide" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')
        lines.append(f'    <line class="guide" x1="{plot_left}" y1="{plot_top}" x2="{plot_right}" y2="{plot_top}"/>')

        if gate_value is not None and gate_value <= scale_max:
            gate_y = _map_y(gate_value, scale_min, scale_max, plot_top, plot_bottom)
            lines.append(f'    <line class="gate" x1="{plot_left}" y1="{gate_y:.1f}" x2="{plot_right}" y2="{gate_y:.1f}"/>')
            lines.append(
                f'    <text class="gate-label" x="{plot_right + 8}" y="{gate_y + 4:.1f}">'
                f"gate {_format_gate_threshold(spec.key, gate_value)}</text>"
            )

        lines.append(f'    <text class="axis" x="{panel_x + 56}" y="{plot_bottom + 4}">{_svg_axis_label(spec.key, scale_min)}</text>')
        lines.append(f'    <text class="axis" x="{panel_x + 56}" y="{plot_top + 4}">{_svg_axis_label(spec.key, scale_max)}</text>')

        if len(xs) > 1:
            path_points = " ".join(
                ("M" if index == 0 else "L") + f"{x:.1f} {y:.1f}"
                for index, (x, y) in enumerate(zip(xs, ys, strict=True))
            )
            lines.append(f'    <path class="trend" d="{path_points}"/>')

        for index, run in enumerate(comparable_runs):
            fill, stroke = SVG_COLORS[index % len(SVG_COLORS)]
            x = xs[index]
            y = ys[index]
            value_label_y = y - 10 if index % 2 == 0 else y + 18
            if value_label_y < plot_top + 10:
                value_label_y = y + 18
            if value_label_y > plot_bottom + 14:
                value_label_y = y - 10
            lines.append(f'    <line class="guide" x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" opacity="0.35"/>')
            lines.append(
                f'    <circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
            lines.append(
                f'    <text class="value" x="{x:.1f}" y="{value_label_y:.1f}" text-anchor="middle">'
                f"{_svg_value_label(spec.key, values[index])}</text>"
            )
            lines.append(
                f'    <text class="tick" x="{x:.1f}" y="{plot_bottom + 24}" text-anchor="middle">'
                f"{run.generated_at.strftime('%m-%d %H:%M')}</text>"
            )
        lines.append("  </g>")
        lines.append("")

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def refresh_history_report(
    *,
    output_root: Path,
    readme_path: Path,
    svg_path: Path,
) -> tuple[StoredRun, list[StoredRun]]:
    runs = load_history_runs(output_root)
    latest_run_id = load_latest_run_id(output_root)
    latest, comparable_runs = select_comparable_runs(runs, latest_run_id=latest_run_id)

    readme_block = build_history_readme_block(
        latest=latest,
        comparable_runs=comparable_runs,
        readme_path=readme_path,
        output_root=output_root,
        svg_path=svg_path,
    )
    svg_content = build_history_svg(comparable_runs)

    readme_text = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(_replace_history_block(readme_text, readme_block), encoding="utf-8")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg_content, encoding="utf-8")
    return latest, comparable_runs
