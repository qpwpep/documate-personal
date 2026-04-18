from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from xml.sax.saxutils import escape

from .history_loader import StoredRun, suite_label
from .readme_renderer import format_gate_threshold, format_metric_value


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
    lines.append('  <title id="title">DocuMate benchmark history</title>')
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
        + escape(
            f"{len(comparable_runs)} comparable {suite_label(comparable_runs[-1].summary.fixtures_path)} runs, 6 key metrics."
        )
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
            xs = [
                plot_left + x_padding + (usable_width * index / (len(comparable_runs) - 1))
                for index in range(len(comparable_runs))
            ]
        ys = [_map_y(value, scale_min, scale_max, plot_top, plot_bottom) for value in values]

        lines.append('  <g transform="translate(0,0)">')
        lines.append(f'    <rect class="panel" x="{panel_x}" y="{panel_y}" width="540" height="230" rx="18"/>')
        lines.append(f'    <text class="metric" x="{panel_x + 26}" y="{panel_y + 34}">{escape(spec.key)}</text>')

        subtitle = spec.direction
        if spec.key == "avg_cost_per_case_usd" and gate_value is not None and gate_value > scale_max:
            subtitle = f"{subtitle}; gate {format_gate_threshold(spec.key, gate_value)} is off chart"
        lines.append(f'    <text class="axis" x="{panel_x + 26}" y="{panel_y + 56}">{escape(subtitle)}</text>')

        lines.append(f'    <line class="guide" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>')
        lines.append(f'    <line class="guide" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')
        lines.append(f'    <line class="guide" x1="{plot_left}" y1="{plot_top}" x2="{plot_right}" y2="{plot_top}"/>')

        if gate_value is not None and gate_value <= scale_max:
            gate_y = _map_y(gate_value, scale_min, scale_max, plot_top, plot_bottom)
            lines.append(f'    <line class="gate" x1="{plot_left}" y1="{gate_y:.1f}" x2="{plot_right}" y2="{gate_y:.1f}"/>')
            lines.append(
                f'    <text class="gate-label" x="{plot_right + 8}" y="{gate_y + 4:.1f}">'
                f"gate {format_gate_threshold(spec.key, gate_value)}</text>"
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
    return format_metric_value(metric_key, value)
