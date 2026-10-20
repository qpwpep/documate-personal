"""보고서가 현재 판정 계약과 계획된 사례 기준 비용 관측을 그대로 표시한다."""
from __future__ import annotations

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
