"""보고서가 현재 판정 계약과 계획된 사례 기준 비용 관측을 그대로 표시한다."""
from __future__ import annotations

import pytest

from src.eval.history_report import refresh_history_report
from src.eval.reporting.markdown import build_markdown_report
from src.eval.reporting.summary import MEASUREMENT_CONTRACT_VERSION
from src.eval.reporting.writer import write_run_outputs
from src.eval.summary_models import RunSummary
from tests.eval.test_release_eval_contract import _case_result, _summary


def test_saved_current_report_displays_release_pass_and_cost_observations(tmp_path):
    pair = _case_result()
    summary = _summary([pair])
    assert summary.overall_passed is True

    write_run_outputs(output_dir=tmp_path, summary=summary, results=[pair[1]])

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert f"- Measurement contract: `{MEASUREMENT_CONTRACT_VERSION}`" in report
    assert "- Release: `PASS`" in report
    assert "legacy_unverified" not in report
    assert "Historical verdict" not in report
    assert "| cost_observed_cases | 1 / 1 |" in report
    assert "| cost_observation_rate | 1.0 |" in report
    assert "llm_call_coverage_rate (historical)" not in report


def test_saved_current_report_uses_planned_cases_for_partial_cost_observation(tmp_path):
    pairs = [_case_result(0), _case_result(1)]
    results = [pairs[0][1]]
    summary = _summary(pairs, results=results)
    assert summary.overall_passed is False
    assert summary.metrics.planned_cases == 2
    assert summary.metrics.total_cases == 1

    write_run_outputs(output_dir=tmp_path, summary=summary, results=results)

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "- Release: `FAIL`" in report
    assert "Historical verdict" not in report
    assert "| cost_observed_cases | 1 / 2 |" in report
    assert "| cost_observation_rate | 0.5 |" in report


def test_historical_report_keeps_historical_verdict_and_cost_coverage():
    payload = _summary([_case_result()]).model_dump(mode="json")
    payload.update(
        decision_contract_version=None,
        measurement_contract_version="llm-usage-scenario-v2",
        case_policy_snapshots={},
        results_fingerprint=None,
        release_decision=None,
    )
    summary = RunSummary.model_validate(payload)

    report = build_markdown_report(
        summary, historical_summary={"metrics": {"llm_call_coverage_rate": 0.75}},
    )

    assert "- Release: `legacy_unverified (not eligible under the current contract)`" in report
    assert "- Historical verdict: `PASS`" in report
    assert "| llm_call_coverage_rate (historical) | 0.75 |" in report
    assert "| cost_observed_cases |" not in report


@pytest.mark.parametrize("test_result", [
    "1239 passed, 120 skipped, 69 subtests passed",
    "1 failed, 7 passed, 2 xfailed in 3.5s",
    "아직 실행하지 않음 — 별도 검증 예정",
])
def test_history_refresh_changes_only_generated_benchmark_content(tmp_path, test_result):
    pair = _case_result()
    summary = _summary([pair])
    output_root = tmp_path / "runs"
    write_run_outputs(output_dir=output_root / summary.run_id, summary=summary, results=[pair[1]])
    prefix = (
        "# Project\n\n## 검증 결과\n\n"
        "테스트 기록은 별도로 검증한 결과입니다.\n\n"
        "| 항목 | 결과 |\n|---|---|\n"
        f"| 테스트 | `{test_result}` |\n\n"
        "<!-- BEGIN BENCHMARK HISTORY -->"
    )
    suffix = (
        "<!-- END BENCHMARK HISTORY -->\n\n"
        "## 요청 계약 검증\n\n기존 계약 평가의 원문 기록.\n\n"
        "## 문서\n\n[문서](docs/)\n"
    )
    readme = tmp_path / "README.md"
    readme.write_text(prefix + "\n\nOld benchmark.\n\n" + suffix, encoding="utf-8")
    svg = tmp_path / "history.svg"

    refresh_history_report(output_root=output_root, readme_path=readme, svg_path=svg)
    updated = readme.read_text(encoding="utf-8")

    assert updated.startswith(prefix)
    assert updated.endswith(suffix)
    assert "Old benchmark." not in updated
    assert summary.run_id in updated
    assert updated.count("| 테스트 |") == 1
    assert svg.is_file()
    refresh_history_report(output_root=output_root, readme_path=readme, svg_path=svg)
    assert readme.read_text(encoding="utf-8") == updated


@pytest.mark.parametrize("markers", [
    "Old history.",
    "<!-- BEGIN BENCHMARK HISTORY -->\nOld history.",
    "<!-- END BENCHMARK HISTORY -->\n<!-- BEGIN BENCHMARK HISTORY -->",
    "<!-- BEGIN BENCHMARK HISTORY -->\n<!-- BEGIN BENCHMARK HISTORY -->\n<!-- END BENCHMARK HISTORY -->",
])
def test_history_refresh_rejects_ambiguous_or_missing_bounds_without_writes(tmp_path, markers):
    pair = _case_result()
    summary = _summary([pair])
    output_root = tmp_path / "runs"
    write_run_outputs(output_dir=output_root / summary.run_id, summary=summary, results=[pair[1]])
    readme = tmp_path / "README.md"
    readme.write_text(f"# Project\n\n## 검증 결과\n{markers}\n\n## 문서\n", encoding="utf-8")
    svg = tmp_path / "history.svg"
    svg.write_text("Original SVG", encoding="utf-8")
    before = (readme.read_bytes(), svg.read_bytes())

    with pytest.raises(ValueError, match="markers"):
        refresh_history_report(output_root=output_root, readme_path=readme, svg_path=svg)

    assert (readme.read_bytes(), svg.read_bytes()) == before
