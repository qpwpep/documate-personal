"""Persisted diagnostics must never masquerade as a current release approval."""

import argparse
from datetime import datetime
import json

import pytest

from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.history_loader import StoredRun
from src.eval.main import command_report
from src.eval.readme_renderer import build_history_readme_block
from src.eval.reporting import build_markdown_report, build_summary
from src.eval.result_models import CaseResult
from src.eval.summary_models import RunSummary
from src.eval.svg_renderer import build_history_svg


def _legacy_payload(**changes):
    return {
        "run_id": "legacy-consumer-run", "endpoint": "http://fixture",
        "fixtures_path": "cases.jsonl", "config_path": "config.toml",
        "generated_at_utc": "2026-01-01T00:00:00+00:00", "track": "release",
        "metrics": {"total_cases": 1, "scored_cases": 1, "passed_cases": 1,
                    "pass_rate": 1.0, "tool_precision": 1.0, "tool_recall": 1.0,
                    "citation_compliance": 1.0},
        "gates": [], "overall_passed": True, "weights": {}, "hard_gates": {},
        "pricing": {}, "judge_enabled": True, "judge_model": "fixture",
        **changes,
    }


def _legacy_result_payload(*, passed=True, judge_state=False):
    payload = {
        "run_id": "legacy-consumer-run", "case_id": "historical-case", "category": "docs_only",
        "query": "Explain arrays", "session_id": "historical-session", "endpoint": "http://fixture",
        "request_payload": {"query": "Explain arrays"}, "http_status": 200,
        "created_at_utc": "2026-01-01T00:00:00+00:00", "passed": passed,
        "final_score": 0.9 if passed else 0.2, "gate_failures": [] if passed else ["historical_quality_floor"],
    }
    if judge_state:
        payload.update(
            judge_status="succeeded", eval_validity="valid", judge_pass=passed,
            llm_judge_score=0.9 if passed else 0.2,
            judge_subscores={name: 0.9 if passed else 0.2 for name in (
                "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language",
            )},
            release_pass=passed, product_pass=passed,
        )
    return payload


def _write_historical_inputs(run_path, *, passed=True, judge_state=False):
    summary = _legacy_payload(overall_passed=passed)
    summary["metrics"].update(passed_cases=int(passed), pass_rate=float(passed))
    if not passed:
        summary["metrics"]["failures"] = [{
            "case_id": "historical-case", "category": "docs_only", "reason": "historical_quality_floor",
        }]
    raw = _legacy_result_payload(passed=passed, judge_state=judge_state)
    (run_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_path / "raw_results.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")
    return summary, raw


def _result(case):
    return CaseResult.model_validate({
        "decision_contract_version": 1,
        "judge_status": "disabled", "eval_validity": "incomplete",
        "policy_snapshot": {"final_forbidden_tools": case.forbidden_tools,
                            "setup_turn_count": len(case.setup_turns)},
        "run_id": "consumer-run", "case_id": case.case_id, "category": case.category,
        "query": case.query, "session_id": "consumer-session", "endpoint": "http://fixture",
        "request_payload": {"query": case.query}, "http_status": 200,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
    })


def test_policy_failure_remains_visible_despite_a_positive_judge_explanation():
    case = BenchmarkCase(case_id="unobserved-case", category="docs_only", query="Explain arrays",
                         forbidden_tools=["slack_notify"])
    result = _result(case)
    result.llm_judge_reason = "The response fully answers the question."
    summary = build_summary(
        run_id="consumer-run", endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="release", requested_limit=None,
        config=BenchmarkConfig(), cases=[case], results=[result],
    )

    assert "tool_execution_evidence_missing" in summary.metrics.failures[0]["reason"]


def _current_run(*, forbidden=False):
    from tests.eval.test_tool_policy_release import _result, _summary

    pair = _result(forbidden=forbidden)
    return _summary([pair]), [pair[1]]


def test_removing_decision_version_cannot_turn_current_fields_into_trusted_history():
    summary, _ = _current_run()
    payload = summary.model_dump(mode="json")
    payload.pop("decision_contract_version")

    with pytest.raises(ValueError, match="(?i)(version|contract|historical|legacy)"):
        RunSummary.model_validate(payload)


def test_current_case_cannot_default_missing_judge_state_to_legacy_success():
    _, results = _current_run()
    payload = results[0].model_dump(mode="json")
    for field in ("judge_status", "llm_judge_score", "judge_subscores", "judge_score_total"):
        payload.pop(field, None)

    with pytest.raises(ValueError, match="(?i)(judge|state|legacy|contract)"):
        CaseResult.model_validate(payload)


def test_current_summary_cannot_default_missing_policy_counts_to_zero():
    summary, _ = _current_run()
    payload = summary.model_dump(mode="json")
    for field in list(payload["metrics"]):
        if field.startswith("policy_"):
            payload["metrics"].pop(field)

    with pytest.raises(ValueError, match="(?i)(policy|count|contract)"):
        RunSummary.model_validate(payload)


@pytest.mark.parametrize("field", ["final_forbidden_tools", "setup_turn_count"])
def test_current_case_requires_explicit_policy_scope_fields(field):
    _, results = _current_run()
    payload = results[0].model_dump(mode="json")
    payload["policy_snapshot"].pop(field)

    with pytest.raises(ValueError, match="(?i)(policy|field|required)"):
        CaseResult.model_validate(payload)
