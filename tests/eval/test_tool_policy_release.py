"""Execution policy must survive scoring, aggregation, and stored-result reads."""
from __future__ import annotations

from argparse import Namespace
from copy import deepcopy

import pytest

from src.eval.config_models import BenchmarkConfig
from src.eval.online_runner.response_parser import parse_agent_response
from src.eval.online_runner.result_builder import build_case_result
from src.eval.reporting.summary import build_summary
from src.eval.result_models import CaseResult, ScenarioTurnResult
from tests.eval.test_release_eval_contract import (
    _cited_answer, _docs_case, _final_payload, _hit, _judge, _judge_payload,
)
from tests.eval.response_fixtures import slack_action, plain_response, execution_evidence


def _evidence(*, forbidden=False, status="complete", phase="succeeded"):
    events = []
    for tool in ["tavily_search", *(["slack_notify"] if forbidden else [])]:
        invocation = f"call-{tool}"
        terminal = phase if tool == "slack_notify" else "succeeded"
        phases = ["blocked"] if terminal == "blocked" else ["started", terminal]
        for item in phases:
            events.append({"sequence": len(events) + 1, "invocation_id": invocation,
                           "tool_name": tool, "phase": item})
    return {"schema_version": 1, "request_id": "req-1", "status": status, "events": events}


def _result(index=0, *, forbidden=False, evidence=True, phase="succeeded", actions=None, debug_overrides=None,
            prior_turns=None, setup_forbidden_tools=None):
    case = _docs_case(case_id=f"case-{index}", forbidden_tools=["slack_notify"],
                      setup_turns=[turn.query for turn in (prior_turns or [])],
                      setup_forbidden_tools=setup_forbidden_tools)
    response, source = _cited_answer("merge는 두 DataFrame의 일치하는 열을 기준으로 행을 결합합니다.")
    response["actions"] = actions or []
    calls = ["tavily_search", *(["slack_notify"] if forbidden and phase != "blocked" else [])]
    payload = _final_payload(response, tool_calls=calls, observed_hits=[_hit(source)],
                             debug_overrides={"execution_evidence": _evidence(forbidden=forbidden, phase=phase)
                                              if evidence else None, **(debug_overrides or {})})
    result = build_case_result(
        run_id="policy-run", endpoint_url="http://fixture", case=case,
        judge=_judge(_judge_payload(1.0)), config=BenchmarkConfig(), session_id=f"session-{index}",
        created_at="2026-09-28T00:00:00+00:00", request_payload={"query": case.query},
        latency_ms_e2e=10, parsed_response=parse_agent_response(payload),
        prior_turns=prior_turns,
    )
    return case, result


def _summary(pairs):
    return build_summary(run_id="policy-run", endpoint="http://fixture", fixtures_path="cases.jsonl",
                         config_path="config.toml", track="release", requested_limit=None,
                         config=BenchmarkConfig(), cases=[case for case, _ in pairs],
                         results=[result for _, result in pairs])


def test_maximum_quality_cannot_offset_forbidden_execution():
    _, result = _result(forbidden=True)
    assert result.composite_quality_score == pytest.approx(1.0)
    assert result.product_pass is True
    assert result.judge_pass is True
    assert result.release_pass is False
    assert "forbidden_tool_execution" in result.gate_failures


def test_nine_good_cases_cannot_dilute_one_policy_violation():
    pairs = [_result(index, forbidden=index == 9) for index in range(10)]
    summary = _summary(pairs)
    assert summary.overall_passed is False
    gates = {gate.name: gate for gate in summary.gates}
    assert gates["release_pass_rate"].passed is True
    assert gates["tool_precision"].passed is True
    assert gates["tool_execution_policy"].passed is False
    assert gates["tool_execution_policy"].actual == 1
    assert "forbidden_tool_execution" in summary.release_decision.failure_codes


def test_case_failure_alone_is_not_a_run_policy_gate():
    pairs = [_result(index, forbidden=index == 9) for index in range(10)]
    pairs[-1][1].release_pass = False
    pairs[-1][1].passed = False
    pairs[-1][1].gate_failures = ["forbidden_tool_execution"]
    summary = _summary(pairs)
    assert summary.overall_passed is False


def test_missing_execution_evidence_blocks_release():
    _, result = _result(evidence=False)
    assert result.release_pass is False
    assert "tool_execution_evidence_missing" in result.gate_failures


@pytest.mark.parametrize("phase", ["failed", "succeeded"])
def test_execution_failure_does_not_erase_forbidden_start(phase):
    _, result = _result(forbidden=True, phase=phase)
    assert result.release_pass is False
    assert "forbidden_tool_execution" in result.gate_failures


def test_blocked_before_invocation_can_pass():
    _, result = _result(forbidden=True, phase="blocked")
    assert result.release_pass is True


def test_complete_authorized_execution_can_pass():
    pair = _result()
    assert pair[1].release_pass is True
    assert _summary([pair]).overall_passed is True


@pytest.mark.parametrize("diagnostics", [None, 1, [{"tool": "tavily_search", "status": []}]])
def test_perfect_quality_cannot_hide_invalid_retrieval_observations(diagnostics):
    """A perfect judge cannot turn malformed current diagnostics into release evidence."""
    pair = _result(debug_overrides={"retrieval_diagnostics": diagnostics})
    result = pair[1]

    assert result.release_pass is False
    assert any("retrieval_diagnostics" in error for error in result.response_errors)
    assert result.policy_assessment.status == "indeterminate"
    assert "tool_execution_retrieval_diagnostics_invalid" in result.policy_assessment.failure_codes
    assert _summary([pair]).overall_passed is False


@pytest.mark.parametrize("defect", ["null", "missing", "normalization_marker", "unmatched"])
def test_retained_final_diagnostics_are_checked_even_when_execution_history_matches(tmp_path, defect):
    from src.eval.main import command_report
    from src.eval.reporting.writer import load_run_outputs, write_run_outputs

    pair = _result()
    result = pair[1]
    final_debug = deepcopy(result.debug)
    if defect == "null":
        final_debug["retrieval_diagnostics"] = None
    elif defect == "missing":
        final_debug.pop("retrieval_diagnostics")
    elif defect == "normalization_marker":
        final_debug["missing_required_debug_fields"] = ["retrieval_diagnostics"]
    else:
        final_debug["retrieval_diagnostics"] = [
            {"tool": "tavily_search", "status": "success", "invocation_id": "unobserved-call"},
        ]
    result.scenario_turns = [ScenarioTurnResult(
        query=result.query, request_payload=result.request_payload, request_id=result.request_id,
        response=result.response, debug=final_debug, execution_evidence=result.execution_evidence,
        tool_calls=result.tool_calls,
    )]

    summary = _summary([pair])

    assert result.policy_assessment.status == "indeterminate"
    expected = ("tool_execution_retrieval_unmatched" if defect == "unmatched"
                else "tool_execution_retrieval_diagnostics_invalid")
    assert expected in result.policy_assessment.failure_codes
    assert summary.overall_passed is False
    write_run_outputs(output_dir=tmp_path, results=[result], summary=summary)
    loaded_summary, loaded_results = load_run_outputs(tmp_path)
    assert loaded_results[0].policy_assessment == result.policy_assessment
    assert loaded_summary.release_decision == summary.release_decision
    assert command_report(Namespace(run=tmp_path)) == 0
    assert expected in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_receipt_cannot_claim_success_without_execution_evidence():
    action = slack_action()
    action["invocation_id"] = "missing-slack-execution"
    _, result = _result(actions=[action])
    assert result.release_pass is False
    assert "tool_execution_receipt_unmatched" in result.gate_failures


def test_malformed_journal_retains_confirmed_forbidden_start():
    raw = _evidence(forbidden=True)
    raw["events"].pop()
    _, result = _result(forbidden=True, debug_overrides={"execution_evidence": raw})
    assert result.release_pass is False
    assert result.policy_assessment.status == "violated"
    assert "forbidden_tool_execution" in result.gate_failures
    assert "tool_execution_evidence_invalid" in result.gate_failures


def test_request_mismatch_blocks_release():
    raw = _evidence()
    raw["request_id"] = "different-request"
    _, result = _result(debug_overrides={"execution_evidence": raw})
    assert result.release_pass is False
    assert "tool_execution_request_mismatch" in result.gate_failures


def test_reused_receipt_is_not_a_new_forbidden_execution():
    raw = _evidence()
    raw["events"].append({"sequence": 3, "invocation_id": "reuse-1", "tool_name": "slack_notify",
                          "phase": "reused", "origin_invocation_id": "setup-req-invocation-1"})
    action = slack_action()
    action["invocation_id"] = "setup-req-invocation-1"
    _, result = _result(actions=[action], debug_overrides={"execution_evidence": raw},
                       prior_turns=[_setup_turn(["slack_notify"])], setup_forbidden_tools=[["save_text"]])
    assert result.release_pass is True


def _setup_turn(tools):
    payload = _final_payload(plain_response("준비했습니다."), tool_calls=tools,
                             debug_overrides={"execution_evidence": execution_evidence(tools, request_id="setup-req")})
    parsed = parse_agent_response(payload, request_id="setup-req")
    return ScenarioTurnResult(query="먼저 준비해줘", request_payload={"query": "먼저 준비해줘"},
                               request_id="setup-req", response=parsed.response, debug=parsed.debug,
                               execution_evidence=parsed.execution_evidence, tool_calls=parsed.tool_calls)


def test_reused_origin_must_be_observable_in_the_scenario():
    raw = _evidence()
    raw["events"].append({"sequence": 3, "invocation_id": "reuse-1", "tool_name": "slack_notify",
                          "phase": "reused", "origin_invocation_id": "unobserved-call"})
    _, result = _result(debug_overrides={"execution_evidence": raw})
    assert result.release_pass is False
    assert "tool_execution_reuse_unverifiable" in result.gate_failures


def test_unplanned_scenario_turn_cannot_be_ignored():
    pair = _result()
    pair[1].scenario_turns = [_setup_turn(["slack_notify"]), _setup_turn([])]
    summary = _summary([pair])
    assert summary.overall_passed is False
    assert "tool_execution_turn_coverage_invalid" in summary.release_decision.failure_codes


def test_final_turn_copy_must_match_the_case_execution_evidence():
    pair = _result()
    pair[1].scenario_turns = [_setup_turn(["slack_notify"])]
    summary = _summary([pair])
    assert summary.overall_passed is False
    assert "tool_execution_final_turn_mismatch" in summary.release_decision.failure_codes


def test_setup_policy_violation_blocks_a_perfect_final_answer():
    _, result = _result(prior_turns=[_setup_turn(["slack_notify"])], setup_forbidden_tools=[["slack_notify"]])
    assert result.release_pass is False
    assert result.policy_assessment.violations[0].turn_index == 0


def test_unspecified_setup_policy_does_not_mean_allow_all():
    _, result = _result(prior_turns=[_setup_turn([])])
    assert result.release_pass is False
    assert "tool_policy_setup_missing" in result.gate_failures


@pytest.mark.parametrize("field", ["events", "schema_version"])
def test_incomplete_envelope_cannot_default_to_complete_empty_history(field):
    raw = _evidence()
    raw.pop(field)
    _, result = _result(debug_overrides={"execution_evidence": raw})
    assert result.release_pass is False
    assert "tool_execution_evidence_invalid" in result.gate_failures


def test_unknown_execution_tool_is_not_assumed_allowed():
    raw = _evidence()
    for phase in ("started", "succeeded"):
        raw["events"].append({"sequence": len(raw["events"]) + 1, "invocation_id": "unknown-call",
                              "tool_name": "unregistered_tool", "phase": phase})
    _, result = _result(debug_overrides={"execution_evidence": raw})
    assert result.release_pass is False
    assert "tool_execution_unknown_tool" in result.gate_failures


def test_current_policy_snapshot_rejects_unknown_prohibited_tool():
    _, result = _result()
    payload = result.model_dump(mode="json")
    payload["policy_snapshot"]["final_forbidden_tools"].append("unregistered_tool")
    with pytest.raises(ValueError, match="unknown execution tool"):
        CaseResult.model_validate(payload)


def test_reassessment_cannot_replace_the_policy_that_detected_a_violation():
    case, result = _result(forbidden=True)
    changed = case.model_copy(update={"forbidden_tools": []})
    with pytest.raises(ValueError, match="policy.*snapshot"):
        _summary([(changed, result)])
    assert result.release_pass is False
    assert result.policy_assessment.status == "violated"


@pytest.mark.parametrize("overrides", [{"tool_calls": ["tavily_search", "slack_notify"]}, {"tool_call_count": 9}])
def test_complete_journal_cannot_hide_conflicting_execution_observations(overrides):
    _, result = _result(debug_overrides=overrides)
    assert result.release_pass is False
    assert "tool_execution_observation_conflict" in result.gate_failures


def test_forbidden_failure_survives_a_successful_retry_with_perfect_quality():
    raw = _evidence(forbidden=True, phase="failed")
    for phase in ("started", "succeeded"):
        raw["events"].append({"sequence": len(raw["events"]) + 1, "invocation_id": "allowed-retry",
                              "tool_name": "tavily_search", "phase": phase})
    pair = _result(forbidden=True, debug_overrides={"execution_evidence": raw, "tool_call_count": 3})
    assert pair[1].release_pass is False
    assert "forbidden_tool_execution" in _summary([pair]).release_decision.failure_codes


def test_forbidden_setup_execution_survives_successful_final_reuse():
    raw = _evidence()
    raw["events"].append({"sequence": 3, "invocation_id": "reuse-1", "tool_name": "slack_notify",
                          "phase": "reused", "origin_invocation_id": "setup-req-invocation-1"})
    pair = _result(prior_turns=[_setup_turn(["slack_notify"])], setup_forbidden_tools=[["slack_notify"]],
                   debug_overrides={"execution_evidence": raw})
    assert pair[1].release_pass is False
    assert "forbidden_tool_execution" in _summary([pair]).release_decision.failure_codes
