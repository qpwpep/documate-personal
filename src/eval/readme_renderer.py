from __future__ import annotations

import os
import re
from pathlib import Path

from .history_loader import StoredRun, latest_run_pointer_name, suite_label
from .summary_models import RunSummary, RunTrack


README_HISTORY_START = "## 검증 결과"
README_HISTORY_END = "## 문서"
HISTORY_TABLE_METRICS = [
    "pass_rate",
    "tool_precision",
    "tool_recall",
    "citation_compliance",
    "p50_latency_ms",
    "p95_latency_ms",
    "avg_cost_per_case_usd",
]
README_TEST_RESULT_PATTERN = re.compile(
    r"^\| [^|]+ \| `(?P<result>\d+ passed, \d+ subtests passed)` \|$",
    re.MULTILINE,
)


def _existing_test_result(readme_path: Path) -> str:
    try:
        readme_text = readme_path.read_text(encoding="utf-8")
    except OSError:
        return "pytest result not recorded"
    match = README_TEST_RESULT_PATTERN.search(readme_text)
    if match is None:
        return "pytest result not recorded"
    return match.group("result")


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
    local_output_root = _relative_markdown_path(output_root, readme_path.parent)
    local_pointer_path = f"{local_output_root}/{pointer_name}"
    local_run_output = f"{local_output_root}/<run_id>"
    test_result = _existing_test_result(readme_path)

    lines: list[str] = []
    lines.append("## 검증 결과")
    lines.append("")
    lines.append(
        f"최신 문서화된 `{track}` benchmark는 `{latest.run_id}` 런입니다. "
        f"로컬 benchmark 실행은 `{local_pointer_path}`를 최신 `{track}` run 포인터로 갱신합니다."
    )
    lines.append("")
    lines.append("| 항목 | 결과 |")
    lines.append("|---|---:|")
    lines.append(f"| 테스트 | `{test_result}` |")
    lines.append(f"| release benchmark | `{latest.metrics.passed_cases}/{latest.metrics.total_cases}` cases passed |")
    lines.append(f"| release pass rate | `{format_metric_value('pass_rate', latest.metrics.pass_rate)}` |")
    lines.append(
        "| tool precision / recall | "
        f"`{format_metric_value('tool_precision', latest.metrics.tool_precision)}` / "
        f"`{format_metric_value('tool_recall', latest.metrics.tool_recall)}` |"
    )
    lines.append(
        f"| citation compliance | `{format_metric_value('citation_compliance', latest.metrics.citation_compliance)}` |"
    )
    lines.append(f"| p95 latency | `{format_metric_value('p95_latency_ms', latest.metrics.p95_latency_ms)} ms` |")
    lines.append(
        f"| avg cost per case | `${format_metric_value('avg_cost_per_case_usd', latest.metrics.avg_cost_per_case_usd)}` |"
    )
    lines.append("")
    lines.append(
        "최신 런은 {passed} Hard Gate를 통과했으며 {failed}는 추가 확인 대상입니다.".format(
            passed=_quoted_metric_names(passed_gates) if passed_gates else "아직 어떤",
            failed=_quoted_metric_names(failed_gates) if failed_gates else "실패한 gate 없음",
        )
    )
    lines.append("")
    if previous is None:
        lines.append(
            f"저장소에 남아 있는 비교 가능한 {current_suite_label} 런은 현재 `{latest.run_id}` 하나뿐입니다."
        )
    else:
        lines.append(
            "비교 가능한 {count}개 {suite} 런 기준으로 최신 `{latest_run}` 런은 직전 `{previous_run}` 대비 "
            "`pass_rate` {pass_rate}, `citation_compliance` {citation}, `p95_latency_ms` {p95_latency} 변화를 보였습니다.".format(
                count=len(comparable_runs),
                suite=current_suite_label,
                latest_run=latest.run_id,
                previous_run=previous.run_id,
                pass_rate=format_delta(
                    "pass_rate",
                    float(latest.metrics.pass_rate) - float(previous.metrics.pass_rate),
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
            )
        )
    lines.append("")

    lines.append(
        f"추세 그래프는 [{svg_markdown_path}]({svg_markdown_path})에 보관합니다. "
        f"실행 방법은 [벤치마크 가이드](docs/benchmarking.md)를 참고하세요. "
        f"로컬 run의 기계 판독 결과와 상세 분석은 각각 `{local_run_output}/summary.json`, "
        f"`{local_run_output}/report.md`에서 확인합니다."
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
