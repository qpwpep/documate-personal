"""Old measurements remain saved facts, even when their decision contract is current."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from src.eval.history_loader import load_history_runs, select_comparable_runs
from src.eval.main import command_report
from src.eval.reporting.writer import load_report_inputs, load_run_outputs


FIXTURE = Path(__file__).parents[1] / "fixtures" / "eval" / "usage_v1_run.json"


def _fingerprint(rows):
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _saved_run(path, *, mutate=None, refresh_fingerprint=False):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(payload)
    if refresh_fingerprint:
        payload["summary"]["results_fingerprint"] = _fingerprint(payload["results"])
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(payload["summary"], ensure_ascii=False), encoding="utf-8")
    (path / "raw_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload["results"]), encoding="utf-8",
    )
    return payload


def test_versioned_old_measurement_is_read_without_current_usage_coercion(tmp_path):
    saved = _saved_run(tmp_path)
    before = {name: (tmp_path / name).read_bytes() for name in ("summary.json", "raw_results.jsonl")}

    inputs = load_report_inputs(tmp_path)

    assert inputs.current_results is None
    assert inputs.historical_results == saved["results"]
    assert inputs.historical_summary == saved["summary"]
    assert inputs.historical_results[0]["token_usage"] == {
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
    }
    assert inputs.historical_results[0]["output_tokens"] == 5
    assert command_report(argparse.Namespace(run=tmp_path)) == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Historical verdict: `PASS`" in report
    assert "legacy_unverified" in report
    assert "llm_call_coverage_rate" in report
    assert "cost_observation_rate" not in report
    assert "- Release: `PASS`" not in report
    for name, expected in before.items():
        assert (tmp_path / name).read_bytes() == expected


def test_current_loader_refuses_old_measurement_even_with_same_decision_version(tmp_path):
    _saved_run(tmp_path)
    with pytest.raises(ValueError, match="(?i)(historical|measurement)"):
        load_run_outputs(tmp_path)


def test_old_raw_fingerprint_is_checked_before_model_conversion(tmp_path):
    def mutate(payload):
        payload["results"][0]["token_usage"]["prompt_tokens"] = 999
    _saved_run(tmp_path, mutate=mutate)

    with pytest.raises(ValueError, match="fingerprint"):
        load_report_inputs(tmp_path)
    assert not (tmp_path / "report.md").exists()


@pytest.mark.parametrize("change", ["other_run", "missing", "duplicate", "policy", "verdict", "evidence"])
def test_old_versioned_results_keep_identity_decision_and_evidence_checks(tmp_path, change):
    def mutate(payload):
        row = payload["results"][0]
        if change == "other_run":
            row["run_id"] = "different-run"
        elif change == "missing":
            payload["results"] = []
        elif change == "duplicate":
            payload["results"].append(deepcopy(row))
        elif change == "policy":
            row["policy_snapshot"]["final_forbidden_tools"] = []
        elif change == "verdict":
            row["decision"]["passed"] = False
        else:
            row["execution_evidence"]["events"] = []
    _saved_run(tmp_path, mutate=mutate, refresh_fingerprint=True)

    with pytest.raises(ValueError, match="(?i)(run|result|case|policy|decision|verdict|evidence|complete)"):
        command_report(argparse.Namespace(run=tmp_path))
    assert not (tmp_path / "report.md").exists()


def test_old_versioned_history_keeps_saved_measurement_and_raw_bytes(tmp_path):
    run_path = tmp_path / "policy-run"
    saved = _saved_run(run_path)
    before = {name: (run_path / name).read_bytes() for name in ("summary.json", "raw_results.jsonl")}

    runs = load_history_runs(tmp_path)
    latest, comparable = select_comparable_runs(runs, track="release")

    assert latest.raw_summary == saved["summary"]
    assert latest.summary.measurement_contract_version == "attachment-question-scenario-v1"
    assert comparable == runs
    for name, expected in before.items():
        assert (run_path / name).read_bytes() == expected


def test_historical_aliases_and_conflicting_metadata_do_not_change_saved_measurements(tmp_path):
    def mutate(payload):
        row = payload["results"][0]
        row["llm_calls"] = [{
            "stage": "synthesis", "attempt": 1, "path": "plain_fallback",
            "usage_metadata": {},
            "response_metadata": {"token_usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        }]
        row["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        row["output_tokens"] = 0
        row["cost_usd"] = 0.0
    saved = _saved_run(tmp_path, mutate=mutate, refresh_fingerprint=True)

    inputs = load_report_inputs(tmp_path)

    assert inputs.historical_results == saved["results"]
    assert inputs.historical_results[0]["output_tokens"] == 0
    assert inputs.historical_results[0]["cost_usd"] == 0.0
    assert inputs.historical_summary["metrics"] == saved["summary"]["metrics"]


def test_new_measurements_start_a_separate_history_baseline(tmp_path):
    from src.eval.reporting.summary import MEASUREMENT_CONTRACT_VERSION

    _saved_run(tmp_path / "policy-run")
    historical = load_history_runs(tmp_path)[0]
    current = replace(historical, summary=historical.summary.model_copy(update={
        "run_id": "current-run", "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION,
    }), raw_summary=None)

    latest, comparable = select_comparable_runs(
        [historical, current], track="release", latest_run_id="current-run",
    )

    assert latest == current
    assert comparable == [current]


def test_old_versioned_readme_and_svg_never_claim_current_release_eligibility(tmp_path):
    from src.eval.readme_renderer import build_history_readme_block
    from src.eval.svg_renderer import build_history_svg

    _saved_run(tmp_path / "policy-run")
    historical = load_history_runs(tmp_path)[0]
    readme = build_history_readme_block(
        track="release", latest=historical, comparable_runs=[historical],
        readme_path=tmp_path / "README.md", output_root=tmp_path, svg_path=tmp_path / "history.svg",
    )
    svg = build_history_svg([historical])

    assert "legacy_unverified" in readme
    assert "historical llm_call_coverage_rate" in readme
    assert "historical PASS; legacy_unverified" in svg
