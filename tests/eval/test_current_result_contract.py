"""Current evaluation records have one vocabulary and explicit verdicts."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.eval.result_models import CaseResult
from src.eval.decisions import raw_results_fingerprint
from src.eval.history_loader import load_history_runs
from src.eval.reporting.writer import load_report_inputs, load_run_outputs
from src.eval.summary_models import RunSummary
from tests.eval.test_release_eval_contract import _case_result, _summary


def test_new_result_and_summary_export_only_canonical_fields():
    pair = _case_result()
    result = pair[1].model_dump(mode="json")
    summary = _summary([pair]).model_dump(mode="json")

    assert not {"judge_gate_passed", "judge_score_total", "invalid_eval", "latency_ms_e2e", "upload_fixture"}.intersection(result)
    assert not {"passed_cases", "pass_rate", "judge_legacy_unknown_cases"}.intersection(summary["metrics"])
    assert "weights" not in summary
    assert "deterministic_direct_usage_rate" not in summary["metrics"]
    assert "deterministic_direct_usage_rate" not in summary["audit_metrics"]
    assert all(gate["name"] != "deterministic_direct_usage_rate" for gate in summary["gates"])
    assert result["release_pass"] is True
    assert summary["metrics"]["release_pass_rate"] == 1.0


@pytest.mark.parametrize("alias", ["judge_gate_passed", "judge_score_total", "invalid_eval", "latency_ms_e2e"])
def test_current_result_rejects_legacy_aliases(alias):
    payload = _case_result()[1].model_dump(mode="json")
    payload[alias] = False

    with pytest.raises(ValidationError):
        CaseResult.model_validate(payload)


def test_current_result_requires_explicit_contract_version():
    payload = _case_result()[1].model_dump(mode="json")
    payload.pop("decision_contract_version")
    for name in ("policy_snapshot", "policy_assessment", "decision", "execution_evidence"):
        payload.pop(name)

    with pytest.raises(ValidationError):
        CaseResult.model_validate(payload)


def test_current_summary_does_not_infer_track_from_limit():
    payload = _summary([_case_result()]).model_dump(mode="json")
    payload.pop("track")

    with pytest.raises(ValidationError):
        RunSummary.model_validate(payload)


def test_current_summary_requires_explicit_policy_observations():
    payload = _summary([_case_result()]).model_dump(mode="json")
    payload["metrics"].pop("policy_violations")

    with pytest.raises(ValidationError):
        RunSummary.model_validate(payload)


@pytest.mark.parametrize("measurement", ["llm-usage-scenario-v3", "llm-usage-scenario-v4", "llm-usage-scenario-v5"])
def test_previous_decision_records_are_preserved_without_reinterpretation(tmp_path, measurement):
    pair = _case_result()
    result = pair[1].model_dump(mode="json")
    result["decision_contract_version"] = 2
    result.update(judge_gate_passed=result["judge_pass"], judge_score_total=result["llm_judge_score"], invalid_eval=False)
    result["latency_ms_e2e"] = result.pop("question_response_ms")
    result["request_id"] = result.pop("execution_id")
    for turn in [result, *result["scenario_turns"]]:
        if "execution_id" in turn:
            turn["request_id"] = turn.pop("execution_id")
        for evidence in (turn["execution_evidence"], turn["debug"]["execution_evidence"]):
            if "execution_id" in evidence:
                evidence["request_id"] = evidence.pop("execution_id")
        if turn["turn_result"] is not None:
            turn["turn_result"]["request_id"] = turn["turn_result"].pop("execution_id")
    summary = _summary([pair]).model_dump(mode="json")
    summary["decision_contract_version"] = 2
    summary["measurement_contract_version"] = measurement
    summary["audit_metrics"]["scoring_contract_version"] = "release-obligations-v9"
    summary["metrics"].update(
        passed_cases=1, pass_rate=1.0, judge_legacy_unknown_cases=0, deterministic_direct_usage_rate=0.0,
    )
    summary["weights"] = None
    summary["results_fingerprint"] = raw_results_fingerprint([result])
    run_dir = tmp_path / summary["run_id"]
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "raw_results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
    original = {path: path.read_bytes() for path in run_dir.iterdir()}

    with pytest.raises(ValueError, match="current.*contracts"):
        load_run_outputs(run_dir)
    assert load_history_runs(tmp_path) == []
    with pytest.raises(ValueError, match="current.*contracts"):
        load_report_inputs(run_dir)
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("tampering", ["fingerprint", "mixed_results", "mixed_measurement"])
def test_history_does_not_hide_damaged_previous_decision_records(tmp_path, tampering):
    pair = _case_result()
    result = pair[1].model_dump(mode="json")
    result["decision_contract_version"] = 2
    summary = _summary([pair]).model_dump(mode="json")
    summary.update(decision_contract_version=2, measurement_contract_version="llm-usage-scenario-v5")
    summary["audit_metrics"]["scoring_contract_version"] = "release-obligations-v9"
    summary["results_fingerprint"] = raw_results_fingerprint([result])
    if tampering == "fingerprint":
        result["cost_usd"] = 123.0
    elif tampering == "mixed_results":
        result["decision_contract_version"] = 3
        summary["results_fingerprint"] = raw_results_fingerprint([result])
    else:
        summary["measurement_contract_version"] = "llm-usage-scenario-v6"
    run_dir = tmp_path / summary["run_id"]
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "raw_results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_history_runs(tmp_path)


def test_previous_location_measurement_is_preserved_without_reloading_its_checks(tmp_path):
    pair = _case_result()
    result = pair[1].model_dump(mode="json")
    summary = _summary([pair]).model_dump(mode="json")
    result["decision_contract_version"] = 3
    summary.update(
        decision_contract_version=3, execution_contract_version="shared-client-scenario-v2",
        measurement_contract_version="llm-usage-scenario-v6",
    )
    summary["audit_metrics"]["scoring_contract_version"] = "release-obligations-v10"
    for scores in [result["rule_scores"], result["effective_weights"], *summary["weight_profiles"].values()]:
        scores["format_language"] = scores.pop("response_availability")
    summary["results_fingerprint"] = raw_results_fingerprint([result])
    run_dir = tmp_path / summary["run_id"]
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "raw_results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
    original = {path: path.read_bytes() for path in run_dir.iterdir()}

    with pytest.raises(ValueError, match="current.*contracts"):
        load_run_outputs(run_dir)
    assert load_history_runs(tmp_path) == []
    assert {path: path.read_bytes() for path in original} == original

    result["cost_usd"] = 123.0
    (run_dir / "raw_results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_history_runs(tmp_path)
