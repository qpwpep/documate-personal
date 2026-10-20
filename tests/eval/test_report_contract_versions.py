"""Saved reports must keep the current release contract and its raw evidence."""
from __future__ import annotations

import json

import pytest

from src.eval.decisions import DECISION_CONTRACT_VERSION, raw_results_fingerprint
from src.eval.history_loader import load_history_runs
from src.eval.history_report import refresh_history_report
from src.eval.reporting.summary import MEASUREMENT_CONTRACT_VERSION, build_summary
from src.eval.reporting.writer import load_report_inputs, load_run_outputs, write_run_outputs
from tests.eval.test_release_eval_contract import _case_result, _config, _summary


def _save_payloads(directory, summary, results):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "raw_results.jsonl").write_text(
        "\n".join(json.dumps(result) for result in results) + "\n", encoding="utf-8",
    )


@pytest.mark.parametrize("measurement", [None, "attachment-question-scenario-v1"])
def test_historical_report_cannot_be_reinterpreted_by_current_report_loader(tmp_path, measurement):
    pair = _case_result()
    summary = _summary([pair]).model_dump(mode="json")
    summary.update(
        decision_contract_version=None, measurement_contract_version=measurement,
        case_policy_snapshots={}, results_fingerprint=None, release_decision=None,
    )
    result = pair[1].model_dump(mode="json")
    for field in ("decision_contract_version", "policy_snapshot", "policy_assessment", "decision", "execution_evidence"):
        result.pop(field)
    _save_payloads(tmp_path, summary, [result])

    with pytest.raises(ValueError, match="current.*contract"):
        load_report_inputs(tmp_path)
    with pytest.raises(ValueError, match="current.*contract"):
        load_run_outputs(tmp_path)


def test_current_report_roundtrip_preserves_decision_and_raw_cost(tmp_path):
    pair = _case_result()
    summary = _summary([pair])
    write_run_outputs(output_dir=tmp_path, results=[pair[1]], summary=summary)

    loaded_summary, results = load_run_outputs(tmp_path)
    report = load_report_inputs(tmp_path)
    assert loaded_summary.decision_contract_version == DECISION_CONTRACT_VERSION
    assert loaded_summary.measurement_contract_version == MEASUREMENT_CONTRACT_VERSION
    assert loaded_summary.release_decision == summary.release_decision
    assert results[0].cost_usd == pair[1].cost_usd
    assert results[0].action_assessments == pair[1].action_assessments
    assert report.current_results == results


def test_unknown_cost_stays_unknown_and_cannot_become_release_pass_on_reload(tmp_path):
    pair = _case_result()
    pair[1].cost_usd = None
    summary = _summary([pair])
    assert not summary.overall_passed
    write_run_outputs(output_dir=tmp_path, results=[pair[1]], summary=summary)

    loaded_summary, results = load_run_outputs(tmp_path)
    assert results[0].cost_usd is None
    assert loaded_summary.metrics.cost_observed_cases == 0
    assert loaded_summary.metrics.cost_observation_rate == 0
    assert not loaded_summary.overall_passed


@pytest.mark.parametrize("field", ["cost_usd", "action_assessments"])
def test_missing_current_decision_evidence_cannot_be_filled_into_a_saved_pass(tmp_path, field):
    pair = _case_result()
    summary = _summary([pair]).model_dump(mode="json")
    raw_results = [pair[1].model_dump(mode="json")]
    raw_results[0].pop(field)
    # Retain a valid byte fingerprint so rejection verifies evidence completeness.
    summary["results_fingerprint"] = raw_results_fingerprint(raw_results)
    _save_payloads(tmp_path, summary, raw_results)

    with pytest.raises(ValueError):
        load_run_outputs(tmp_path)
    with pytest.raises(ValueError):
        load_report_inputs(tmp_path)


@pytest.mark.parametrize("decision, measurement", [
    (1, "llm-usage-scenario-v2"),
    (DECISION_CONTRACT_VERSION, "future-unknown"),
])
def test_incompatible_version_is_rejected_before_current_projection(tmp_path, decision, measurement):
    pair = _case_result()
    summary = _summary([pair]).model_dump(mode="json")
    summary.update(decision_contract_version=decision, measurement_contract_version=measurement)
    raw_results = [pair[1].model_dump(mode="json")]
    raw_results[0]["decision_contract_version"] = decision
    summary["results_fingerprint"] = raw_results_fingerprint(raw_results)
    _save_payloads(tmp_path, summary, raw_results)

    with pytest.raises(ValueError, match="current.*contract"):
        load_report_inputs(tmp_path)


def test_history_summary_cannot_bypass_current_evidence_validation(tmp_path):
    pair = _case_result()
    summary = _summary([pair]).model_dump(mode="json")
    summary.update(
        decision_contract_version=None, case_policy_snapshots={},
        results_fingerprint=None, release_decision=None,
    )
    _save_payloads(tmp_path / "run", summary, [pair[1].model_dump(mode="json")])

    with pytest.raises(ValueError, match="current.*contract"):
        load_history_runs(tmp_path)


@pytest.mark.parametrize("decision, measurement", [
    (None, None),
    (None, "attachment-question-scenario-v1"),
    (1, "attachment-question-scenario-v1"),
    (1, "llm-usage-scenario-v2"),
])
def test_history_refresh_ignores_historical_runs_without_rewriting_them(tmp_path, decision, measurement):
    pair = _case_result()
    summary = _summary([pair])
    output_root = tmp_path / "runs"
    current = output_root / summary.run_id
    write_run_outputs(output_dir=current, results=[pair[1]], summary=summary)
    historical = output_root / "historical-run"
    historical.mkdir()
    old_summary = summary.model_dump(mode="json")
    old_summary.update(
        run_id=historical.name, decision_contract_version=decision,
        measurement_contract_version=measurement,
    )
    old_summary.pop("release_suite")
    if decision is None:
        for field in ("case_policy_snapshots", "results_fingerprint", "release_decision"):
            old_summary.pop(field)
        # Old summary-only runs must not require raw results in the new contract.
        (historical / "summary.json").write_text(json.dumps(old_summary), encoding="utf-8")
    else:
        old_result = pair[1].model_dump(mode="json")
        old_result.update(run_id=historical.name, decision_contract_version=decision)
        old_summary["results_fingerprint"] = raw_results_fingerprint([old_result])
        _save_payloads(historical, old_summary, [old_result])
    (historical / "report.md").write_text("Original historical report.\n", encoding="utf-8")
    (output_root / "latest_release_run.txt").write_text(historical.name, encoding="utf-8")
    before = {path: path.read_bytes() for path in output_root.rglob("*") if path.is_file()}
    readme = tmp_path / "README.md"
    readme.write_text("# Fixture\n\n## 검증 결과\n\nOld history.\n\n## 문서\n", encoding="utf-8")
    svg = tmp_path / "history.svg"

    latest, comparable = refresh_history_report(
        output_root=output_root, readme_path=readme, svg_path=svg,
    )

    assert latest.run_id == summary.run_id
    assert [run.run_id for run in comparable] == [summary.run_id]
    assert [run.run_id for run in load_history_runs(output_root)] == [summary.run_id]
    assert summary.run_id in readme.read_text(encoding="utf-8")
    assert summary.run_id in svg.read_text(encoding="utf-8")
    assert {path: path.read_bytes() for path in before} == before


def test_history_does_not_skip_corrupt_current_results(tmp_path):
    pair = _case_result()
    summary = _summary([pair])
    directory = tmp_path / summary.run_id
    write_run_outputs(output_dir=directory, results=[pair[1]], summary=summary)
    raw_path = directory / "raw_results.jsonl"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["cost_usd"] = 100
    raw_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_history_runs(tmp_path)


@pytest.mark.parametrize("versions", ["removed", "downgraded"])
def test_history_does_not_skip_current_evidence_with_replaced_versions(tmp_path, versions):
    pair = _case_result()
    summary = _summary([pair]).model_dump(mode="json")
    if versions == "removed":
        summary.pop("decision_contract_version")
        summary.pop("measurement_contract_version")
    else:
        summary.update(decision_contract_version=1, measurement_contract_version="llm-usage-scenario-v2")
    _save_payloads(tmp_path / "run", summary, [pair[1].model_dump(mode="json")])

    with pytest.raises(ValueError, match="current.*contract"):
        load_history_runs(tmp_path)


@pytest.mark.parametrize("decision, measurement", [
    (DECISION_CONTRACT_VERSION, "llm-usage-scenario-v2"),
    (1, MEASUREMENT_CONTRACT_VERSION),
    (DECISION_CONTRACT_VERSION, "future-unknown"),
    (999, "llm-usage-scenario-v2"),
])
def test_history_does_not_skip_mixed_or_unknown_contracts(tmp_path, decision, measurement):
    pair = _case_result()
    summary = _summary([pair]).model_dump(mode="json")
    summary.update(decision_contract_version=decision, measurement_contract_version=measurement)
    _save_payloads(tmp_path / "run", summary, [pair[1].model_dump(mode="json")])

    with pytest.raises(ValueError, match="current.*contract"):
        load_history_runs(tmp_path)


def _write_smoke_history_run(output_root):
    case, result = _case_result()
    result.run_id = "current-smoke"
    summary = build_summary(
        run_id=result.run_id, endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="smoke", requested_limit=1,
        config=_config(), cases=[case], results=[result],
    )
    write_run_outputs(output_dir=output_root / summary.run_id, results=[result], summary=summary)
    return summary


@pytest.mark.parametrize("pointer_target", ["historical-release", "current-smoke", None])
def test_release_history_rejects_smoke_after_filtering_old_release_without_writes(tmp_path, pointer_target):
    output_root = tmp_path / "runs"
    _write_smoke_history_run(output_root)
    pair = _case_result()
    old_summary = _summary([pair]).model_dump(mode="json")
    old_summary.update(
        run_id="historical-release", decision_contract_version=1,
        measurement_contract_version="llm-usage-scenario-v2",
    )
    old_summary.pop("release_suite")
    old_result = pair[1].model_dump(mode="json")
    old_result.update(run_id="historical-release", decision_contract_version=1)
    old_summary["results_fingerprint"] = raw_results_fingerprint([old_result])
    _save_payloads(output_root / "historical-release", old_summary, [old_result])
    if pointer_target is not None:
        (output_root / "latest_release_run.txt").write_text(pointer_target, encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text("# Fixture\n\n## 검증 결과\n\nOriginal release history.\n\n## 문서\n", encoding="utf-8")
    svg = tmp_path / "history.svg"
    svg.write_text("<svg><!-- Original release history --></svg>\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    # The default release history must not fall back to a diagnostic run,
    # regardless of a stale, misdirected, or absent latest-run pointer.
    with pytest.raises(ValueError, match="No release benchmark summaries"):
        refresh_history_report(output_root=output_root, readme_path=readme, svg_path=svg)

    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("track", ["release", "smoke"])
def test_history_ignores_pointer_to_another_track_when_matching_runs_exist(tmp_path, track):
    output_root = tmp_path / "runs"
    smoke = _write_smoke_history_run(output_root)
    pair = _case_result()
    release = _summary([pair])
    write_run_outputs(output_dir=output_root / release.run_id, results=[pair[1]], summary=release)
    expected, other = (release, smoke) if track == "release" else (smoke, release)
    (output_root / f"latest_{track}_run.txt").write_text(other.run_id, encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# Fixture\n\n## 검증 결과\n\nOld history.\n\n## 문서\n", encoding="utf-8")
    svg = tmp_path / "history.svg"

    latest, comparable = refresh_history_report(
        output_root=output_root, readme_path=readme, svg_path=svg, track=track,
    )

    assert latest.track == track
    assert latest.run_id == expected.run_id
    assert [run.run_id for run in comparable] == [expected.run_id]
    assert f"`{track}` benchmark는 `{expected.run_id}`" in readme.read_text(encoding="utf-8")
    assert expected.run_id in svg.read_text(encoding="utf-8")
