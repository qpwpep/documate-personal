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
from src.eval.reporting.writer import load_report_inputs, load_run_outputs, write_run_outputs
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


@pytest.mark.parametrize("passed", [False, True])
@pytest.mark.parametrize("judge_state", [False, True])
def test_report_regenerates_historical_results_without_rejudging_or_rewriting_them(tmp_path, passed, judge_state):
    _, original_raw = _write_historical_inputs(tmp_path, passed=passed, judge_state=judge_state)
    source_bytes = {name: (tmp_path / name).read_bytes() for name in ("summary.json", "raw_results.jsonl")}
    (tmp_path / "report.md").write_text("Old historical report", encoding="utf-8")

    assert command_report(argparse.Namespace(run=tmp_path)) == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert "legacy_unverified" in report
    assert f"Historical verdict: `{'PASS' if passed else 'FAIL'}`" in report
    assert f"| passed_cases | {int(passed)} |" in report
    assert f"| pass_rate | {float(passed)} |" in report
    assert "- Release: `PASS`" not in report
    assert "tool_policy_snapshot_missing" not in report
    assert "Tool Execution Policy" not in report
    if not passed:
        assert "historical_quality_floor" in report
    historical_inputs = load_report_inputs(tmp_path)
    assert historical_inputs.historical_results == [original_raw]
    assert historical_inputs.current_results is None
    assert command_report(argparse.Namespace(run=tmp_path)) == 0
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == report
    assert {name: (tmp_path / name).read_bytes() for name in source_bytes} == source_bytes


def test_report_preserves_a_historical_run_that_recorded_duplicate_results_as_a_failure(tmp_path):
    summary, raw = _write_historical_inputs(tmp_path)
    summary["overall_passed"] = False
    summary["metrics"].update(total_cases=2, scored_cases=2, passed_cases=2, duplicate_result_cases=1)
    summary["gates"] = [{
        "name": "evaluation_completeness", "gate_type": "release", "threshold": 0,
        "actual": 1, "passed": False,
    }]
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "raw_results.jsonl").write_text((json.dumps(raw) + "\n") * 2, encoding="utf-8")
    source_bytes = {name: (tmp_path / name).read_bytes() for name in ("summary.json", "raw_results.jsonl")}

    assert command_report(argparse.Namespace(run=tmp_path)) == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert "legacy_unverified" in report
    assert "Historical verdict: `FAIL`" in report
    assert "| duplicate_result_cases | 1 |" in report
    assert "| passed_cases | 2 |" in report
    assert "| evaluation_completeness | release | 0 | 1 | N |" in report
    assert "- Release: `PASS`" not in report
    assert load_report_inputs(tmp_path).historical_results == [raw, raw]
    assert {name: (tmp_path / name).read_bytes() for name in source_bytes} == source_bytes


@pytest.mark.parametrize("change", [
    "invalid_json", "invalid_shape", "missing_field", "other_run", "duplicate_case", "wrong_duplicate_count",
    "wrong_count",
    "current_raw", "current_summary", "unknown_summary_version", "unknown_raw_version",
    "downgraded_summary", "downgraded_raw",
])
def test_report_refuses_inconsistent_historical_inputs_without_overwriting_report(tmp_path, change):
    summary, raw = _write_historical_inputs(tmp_path)
    rows = [raw]
    if change == "other_run":
        raw["run_id"] = "another-run"
    elif change == "missing_field":
        raw.pop("case_id")
    elif change == "duplicate_case":
        rows.append(dict(raw))
        summary["metrics"]["total_cases"] = 2
    elif change == "wrong_duplicate_count":
        summary["metrics"]["duplicate_result_cases"] = 1
    elif change == "wrong_count":
        summary["metrics"]["total_cases"] = 2
    elif change == "current_raw":
        _, current = _current_run()
        rows = [current[0].model_dump(mode="json")]
        rows[0]["run_id"] = summary["run_id"]
    elif change == "current_summary":
        current_summary, _ = _current_run()
        summary = current_summary.model_dump(mode="json")
        raw["run_id"] = summary["run_id"]
    elif change == "unknown_summary_version":
        summary["decision_contract_version"] = 999
    elif change == "unknown_raw_version":
        raw["decision_contract_version"] = 999
    elif change == "downgraded_summary":
        summary["release_decision"] = {"passed": True, "failure_codes": [], "scope": "release"}
    elif change == "downgraded_raw":
        raw["policy_snapshot"] = {"final_forbidden_tools": [], "setup_turn_count": 0}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    raw_text = "{not json\n" if change == "invalid_json" else (
        "[]\n" if change == "invalid_shape" else "".join(json.dumps(row) + "\n" for row in rows)
    )
    (tmp_path / "raw_results.jsonl").write_text(raw_text, encoding="utf-8")
    (tmp_path / "report.md").write_text("Keep historical report", encoding="utf-8")

    with pytest.raises(ValueError):
        command_report(argparse.Namespace(run=tmp_path))

    assert (tmp_path / "report.md").read_text(encoding="utf-8") == "Keep historical report"


def test_report_still_requires_historical_raw_results(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(_legacy_payload()), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="raw_results"):
        command_report(argparse.Namespace(run=tmp_path))


def test_current_output_loader_does_not_accept_historical_report_inputs(tmp_path):
    _write_historical_inputs(tmp_path)

    with pytest.raises(ValueError, match="historical results cannot establish current release eligibility"):
        load_run_outputs(tmp_path)


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


def test_legacy_pass_is_historical_and_not_a_current_release_verdict():
    summary = RunSummary.model_validate(_legacy_payload())

    report = build_markdown_report(summary)

    assert "- Release: `PASS`" not in report
    assert "legacy_unverified" in report
    assert "Historical verdict: `PASS`" in report


def test_failed_gate_and_stored_pass_cannot_render_current_release_pass():
    summary = RunSummary.model_validate(_legacy_payload(gates=[{
        "name": "forbidden_execution", "gate_type": "release", "threshold": 0,
        "actual": 1, "passed": False,
    }]))

    report = build_markdown_report(summary)

    assert "- Release: `PASS`" not in report


def test_judge_enabled_smoke_does_not_render_a_release_verdict():
    summary = RunSummary.model_validate(_legacy_payload(track="smoke"))

    report = build_markdown_report(summary)

    assert "- Release: `PASS`" not in report
    assert "diagnostic" in report


def test_report_refuses_results_that_do_not_belong_to_the_saved_summary(tmp_path):
    case = BenchmarkCase(case_id="expected-case", category="docs_only", query="Explain arrays")
    result = _result(case)
    summary = build_summary(
        run_id="consumer-run", endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="release", requested_limit=None,
        config=BenchmarkConfig(), cases=[case], results=[result],
    )
    (tmp_path / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    different = result.model_dump(mode="json")
    different["case_id"] = "unrelated-case"
    (tmp_path / "raw_results.jsonl").write_text(json.dumps(different) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)(result|snapshot|case|decision)"):
        command_report(argparse.Namespace(run=tmp_path))

    assert not (tmp_path / "report.md").exists()


def test_legacy_history_labels_historical_pass_without_current_eligibility(tmp_path):
    summary = RunSummary.model_validate(_legacy_payload())
    stored = StoredRun(summary, datetime.fromisoformat(summary.generated_at_utc), track_explicit=True)

    readme = build_history_readme_block(
        track="release", latest=stored, comparable_runs=[stored],
        readme_path=tmp_path / "README.md", output_root=tmp_path / "runs", svg_path=tmp_path / "history.svg",
    )
    svg = build_history_svg([stored])

    assert "legacy_unverified" in readme
    assert "legacy_unverified" in svg
    assert "historical PASS" in svg


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


@pytest.mark.parametrize("gate_change", ["remove", "audit"])
def test_report_refuses_removing_or_downgrading_the_policy_gate(tmp_path, gate_change):
    summary, results = _current_run(forbidden=True)
    write_run_outputs(output_dir=tmp_path, results=results, summary=summary)
    original_report = (tmp_path / "report.md").read_bytes()
    payload = summary.model_dump(mode="json")
    if gate_change == "remove":
        payload["gates"] = [gate for gate in payload["gates"] if gate["name"] != "tool_execution_policy"]
    else:
        next(gate for gate in payload["gates"] if gate["name"] == "tool_execution_policy")["gate_type"] = "audit"
    (tmp_path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)policy"):
        command_report(argparse.Namespace(run=tmp_path))

    assert (tmp_path / "report.md").read_bytes() == original_report


def test_removing_decision_version_cannot_turn_current_fields_into_trusted_history():
    summary, _ = _current_run()
    payload = summary.model_dump(mode="json")
    payload.pop("decision_contract_version")

    with pytest.raises(ValueError, match="(?i)(version|contract|historical|legacy)"):
        RunSummary.model_validate(payload)


def test_current_history_requires_the_original_raw_results(tmp_path):
    from src.eval.history_loader import load_history_runs

    summary, results = _current_run()
    run_dir = tmp_path / summary.run_id
    write_run_outputs(output_dir=run_dir, results=results, summary=summary)
    raw = results[0].model_dump(mode="json")
    raw["query"] = "Changed after evaluation"
    (run_dir / "raw_results.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)(fingerprint|result)"):
        load_history_runs(tmp_path)


@pytest.mark.parametrize("forbidden", [False, True])
def test_report_roundtrip_keeps_the_central_verdict_and_policy_reason(tmp_path, forbidden):
    summary, results = _current_run(forbidden=forbidden)
    write_run_outputs(output_dir=tmp_path, results=results, summary=summary)

    assert command_report(argparse.Namespace(run=tmp_path)) == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert f"- Release: `{'FAIL' if forbidden else 'PASS'}`" in report
    if forbidden:
        assert "forbidden_tool_execution" in report
        assert "slack_notify" in report
        assert "call-slack_notify" in report


def test_history_renderers_reject_a_mutated_policy_gate(tmp_path):
    summary, _ = _current_run()
    next(gate for gate in summary.gates if gate.name == "tool_execution_policy").gate_type = "audit"
    stored = StoredRun(summary, datetime.fromisoformat(summary.generated_at_utc), track_explicit=True)

    with pytest.raises(ValueError, match="(?i)policy"):
        build_history_readme_block(
            track="release", latest=stored, comparable_runs=[stored], readme_path=tmp_path / "README.md",
            output_root=tmp_path / "runs", svg_path=tmp_path / "history.svg",
        )
    with pytest.raises(ValueError, match="(?i)policy"):
        build_history_svg([stored])


@pytest.mark.parametrize("change", ["missing", "duplicate", "other_run", "extra_policy"])
def test_report_rechecks_run_membership_even_with_a_matching_results_fingerprint(tmp_path, change):
    from src.eval.decisions import results_fingerprint

    summary, results = _current_run()
    payload = summary.model_dump(mode="json")
    if change == "missing":
        results = []
    elif change == "duplicate":
        results = results + results
    elif change == "other_run":
        results[0].run_id = "unrelated-run"
    else:
        payload["case_policy_snapshots"]["unexecuted-case"] = dict(next(iter(payload["case_policy_snapshots"].values())))
    payload["results_fingerprint"] = results_fingerprint(results)
    (tmp_path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "raw_results.jsonl").write_text(
        "".join(result.model_dump_json() + "\n" for result in results), encoding="utf-8",
    )

    with pytest.raises(ValueError, match="(?i)(result|snapshot|case|run|complete)"):
        command_report(argparse.Namespace(run=tmp_path))

    assert not (tmp_path / "report.md").exists()


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_incomplete_run_cannot_claim_a_passing_completeness_gate(tmp_path, change):
    from src.eval.decisions import results_fingerprint

    summary, results = _current_run()
    payload = summary.model_dump(mode="json")
    results = [] if change == "missing" else results + results
    payload["results_fingerprint"] = results_fingerprint(results)
    payload["metrics"].update(
        total_cases=len(results), policy_compliant_cases=len(results),
        missing_result_cases=int(change == "missing"), duplicate_result_cases=int(change == "duplicate"),
    )
    (tmp_path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "raw_results.jsonl").write_text(
        "".join(result.model_dump_json() + "\n" for result in results), encoding="utf-8",
    )

    with pytest.raises(ValueError, match="(?i)(complete|run|result)"):
        command_report(argparse.Namespace(run=tmp_path))


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
