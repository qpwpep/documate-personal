"""Release scope and cost evidence are prerequisites, independent of quality."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.contracts.usage import LLMCallRecord, TokenUsage
from src.eval.config_models import BenchmarkConfig
from src.eval.decisions import validate_run_outputs
from src.eval.main import resolve_run_track
from src.eval.online_runner.case_runner import run_online_benchmark
from src.eval.pricing import compute_cost_usd
from src.eval.reporting.summary import build_summary
from src.eval.summary_models import ReleaseSuiteContract, RunSummary
from tests.eval.test_release_eval_contract import _case_result, _config


def _summary(pairs, *, results=None, **kwargs):
    return build_summary(
        run_id="release-contract", endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="release", requested_limit=None,
        config=_config(), cases=[case for case, _ in pairs],
        results=[result for _, result in pairs] if results is None else results,
        **kwargs,
    )


def _suite(pairs):
    return ReleaseSuiteContract(
        approved_case_ids=tuple(case.case_id for case, _ in pairs),
        review_sha256="1" * 64, candidate_sha256="2" * 64,
    )


@pytest.mark.parametrize("limit", [-1, 0, 1, 10000])
def test_explicit_release_rejects_every_limit(limit):
    with pytest.raises(ValueError, match="[Rr]elease.*limit"):
        resolve_run_track("release", limit)


def test_release_limit_rejected_before_loading_or_execution(tmp_path):
    # Nonexistent inputs prove rejection precedes filesystem, judge, and HTTP work.
    with pytest.raises(ValueError, match="[Rr]elease.*limit"):
        run_online_benchmark(
            fixtures_path=tmp_path / "absent.jsonl", endpoint="http://must-not-call",
            config=BenchmarkConfig(), config_path=Path("absent.toml"),
            output_root=tmp_path, track="release", limit=1,
        )


def test_passing_selected_cases_cannot_establish_an_approved_release():
    summary = _summary([_case_result()])

    assert summary.overall_passed is False
    assert "release_suite_completeness" in summary.release_decision.failure_codes


@pytest.mark.parametrize("selection", ["subset", "replacement"])
def test_selected_case_identities_must_match_the_approved_full_suite(selection):
    approved = [_case_result(index) for index in range(2)]
    selected = approved[:1] if selection == "subset" else [_case_result(100), approved[1]]
    summary = _summary(selected, release_suite=_suite(approved))

    assert summary.metrics.release_pass_rate == 1.0
    assert summary.overall_passed is False
    assert next(g for g in summary.gates if g.name == "release_suite_completeness").status == "evaluation_incomplete"


def test_complete_approved_suite_and_observed_cost_can_release():
    pairs = [_case_result(index) for index in range(2)]
    summary = _summary(pairs, release_suite=_suite(pairs))

    assert summary.overall_passed is True
    assert RunSummary.model_validate(summary.model_dump(mode="json")).overall_passed is True


def test_unapproved_suite_is_rejected_before_judge_or_scenario(monkeypatch, tmp_path):
    from src.eval import release_dataset
    from src.eval.online_runner import case_runner

    monkeypatch.setattr(release_dataset, "load_release_input", lambda *args, **kwargs: ([_case_result()[0]], None))

    def forbidden(*args, **kwargs):
        pytest.fail("unapproved input reached model or scenario execution")

    monkeypatch.setattr(case_runner, "LLMJudge", forbidden)
    monkeypatch.setattr(case_runner, "_run_single_case", forbidden)
    with pytest.raises(ValueError, match="approved complete suite"):
        run_online_benchmark(
            fixtures_path=tmp_path / "candidate.jsonl", endpoint="http://must-not-call",
            config=BenchmarkConfig(), config_path=Path("absent.toml"),
            output_root=tmp_path, track="release",
        )


def test_unknown_cost_blocks_release_without_relabeling_product_quality():
    case, result = _case_result()
    result.cost_usd = compute_cost_usd(
        llm_calls=[LLMCallRecord(
            stage="post_synthesis_validation", path="decisions", model_name="unpriced-model",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )],
        pricing=BenchmarkConfig().pricing,
    )
    summary = _summary([(case, result)], release_suite=_suite([(case, result)]))
    gates = {gate.name: gate for gate in summary.gates}

    assert result.cost_usd is None
    assert result.product_pass is True
    assert result.eval_validity == "valid"
    assert summary.metrics.cost_observation_rate == 0.0
    assert gates["avg_cost_per_case_usd"].passed is False
    assert gates["avg_cost_per_case_usd"].status == "observation_insufficient"
    assert summary.overall_passed is False


@pytest.mark.parametrize("observed,eligible", [(7, False), (8, True)])
def test_existing_observation_threshold_is_enforced(observed, eligible):
    pairs = [_case_result(index) for index in range(10)]
    for _, result in pairs[observed:]:
        result.cost_usd = None
    summary = _summary(pairs, release_suite=_suite(pairs))
    gates = {gate.name: gate for gate in summary.gates}

    assert gates["cost_observation_rate"].threshold == 0.8
    assert gates["cost_observation_rate"].passed is eligible
    assert gates["avg_cost_per_case_usd"].passed is eligible
    assert summary.overall_passed is eligible


@pytest.mark.parametrize("gate_name", ["release_suite_completeness", "cost_observation_rate", "avg_cost_per_case_usd"])
def test_saved_summary_cannot_promote_unverified_evidence_to_a_pass(gate_name):
    pairs = [_case_result()]
    pairs[0][1].cost_usd = None
    payload = _summary(pairs).model_dump(mode="json")
    for gate in payload["gates"]:
        if gate["name"] == gate_name:
            gate["passed"] = True

    with pytest.raises(ValueError):
        RunSummary.model_validate(payload)


def test_saved_cost_summary_must_match_the_raw_result_cost_evidence():
    pairs = [_case_result()]
    pairs[0][1].cost_usd = None
    payload = _summary(pairs, release_suite=_suite(pairs)).model_dump(mode="json")
    payload["metrics"].update(
        cost_observed_cases=1, cost_observation_rate=1.0, cost_gate_eligible=True,
        avg_cost_per_case_usd=0.001,
    )
    for gate in payload["gates"]:
        if gate["name"] == "cost_observation_rate":
            gate.update(actual=1.0, passed=True, status="evaluated")
        elif gate["name"] == "avg_cost_per_case_usd":
            gate.update(actual=0.001, passed=True, status="evaluated")
    payload["release_decision"].update(passed=True, failure_codes=[])
    payload["overall_passed"] = True
    summary = RunSummary.model_validate(payload)

    with pytest.raises(ValueError, match="cost.*observ"):
        validate_run_outputs(summary, [result for _, result in pairs])


def test_cost_observation_denominator_includes_missing_planned_results():
    pairs = [_case_result(index) for index in range(10)]
    summary = _summary(pairs, results=[result for _, result in pairs[:5]])

    assert summary.metrics.cost_observation_rate == 0.5
    assert summary.metrics.cost_gate_eligible is False
    assert next(g for g in summary.gates if g.name == "cost_observation_rate").passed is False
