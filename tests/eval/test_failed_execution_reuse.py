"""Observed failed executions remain valid origins without becoming successes."""
from __future__ import annotations

import argparse
from copy import deepcopy

import pytest

from src.core.contracts import PlannerState
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.judge_llm import LLMJudge
from src.eval.main import command_report
from src.eval.online_runner.response_parser import parse_agent_response
from src.eval.online_runner.result_builder import build_case_result
from src.eval.reporting.summary import build_summary
from src.eval.reporting.writer import load_run_outputs, write_run_outputs
from src.eval.result_models import ScenarioTurnResult
from src.infra.tools.local_rag import build_upload_search_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from tests.core.test_actions_nodes import _contract, _state
from tests.core.test_pending_action_delivery import delivery_tools
from tests.core.test_tool_execution_evidence import _manager
from tests.eval.response_fixtures import plain_response
from tests.eval.test_release_eval_contract import _final_payload


def _evaluate(payloads, *, setup_forbidden=None, final_forbidden=(), slack_required=False, retained_turns=None):
    parsed = [parse_agent_response(payload) for payload in payloads]
    turns = [ScenarioTurnResult(
        query="Prepare the result", request_payload={"query": "Prepare the result"},
        request_id=item.request_id, response=item.response, debug=item.debug,
        tool_calls=item.tool_calls, execution_evidence=item.execution_evidence,
    ) for item in parsed[:-1]]
    if retained_turns is not None:
        turns = retained_turns
    case = BenchmarkCase(
        case_id="reuse-case", category="tool_action", query="Continue the same task",
        setup_turns=[turn.query for turn in turns],
        setup_forbidden_tools=setup_forbidden if setup_forbidden is not None else [[] for _ in turns],
        forbidden_tools=list(final_forbidden),
    )
    config = BenchmarkConfig(judge_enabled=False)
    result = build_case_result(
        run_id="reuse-run", endpoint_url="http://fixture", case=case,
        judge=LLMJudge(model_name="unused", enabled=False), config=config,
        session_id="reuse-session", created_at="2026-09-28T00:00:00+00:00",
        request_payload={"query": case.query}, latency_ms_e2e=1,
        parsed_response=parsed[-1], prior_turns=turns, slack_delivery_required=slack_required,
    )
    summary = build_summary(
        run_id="reuse-run", endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="smoke", requested_limit=None,
        config=config, cases=[case], results=[result],
    )
    return result, summary


def _roundtrip(tmp_path, result, summary):
    write_run_outputs(output_dir=tmp_path, results=[result], summary=summary)
    loaded_summary, loaded_results = load_run_outputs(tmp_path)
    assert loaded_results[0].policy_assessment == result.policy_assessment
    assert loaded_results[0].execution_evidence == result.execution_evidence
    assert loaded_summary.release_decision == summary.release_decision
    assert command_report(argparse.Namespace(run=tmp_path)) == 0
    assert "tool_execution_reuse_unverifiable" not in (tmp_path / "report.md").read_text(encoding="utf-8")
    return loaded_results[0]


def _runtime_payload(response):
    # The HTTP boundary normally supplies the recorder's request identity.
    response = deepcopy(response)
    response["trace"] = "Request ID: " + response["debug"]["execution_evidence"]["request_id"]
    return response


def test_actual_failed_search_reuse_is_observed_and_remains_unavailable(tmp_path):
    def prepare(_):
        state = _state("Unavailable uploads", contract=_contract())
        state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[
            RetrievalTask(route="upload", query="alpha", k=1),
        ]))
        return state

    retrieve = make_retrieve_dispatch_node(None, build_upload_search_tool(), False)

    def retrieve_twice(state):
        first = retrieve(state)
        return retrieve({**state, **first})

    manager = _manager(prepare=prepare, action=retrieve_twice)
    payload = _runtime_payload(manager.run_agent_flow("Search uploads once"))
    events = payload["debug"]["execution_evidence"]["events"]
    assert [event["phase"] for event in events] == ["started", "failed", "reused"]
    assert events[-1]["origin_invocation_id"] == events[0]["invocation_id"]

    result, summary = _evaluate([payload])

    assert result.policy_assessment.status == "compliant"
    assert result.tool_calls == ["upload_search"]
    assert result.tool_call_count == 1
    assert result.retrieval_diagnostics[-1].status == "unavailable"
    assert result.retrieval_diagnostics[-1].invocation_id == events[0]["invocation_id"]
    assert next(gate for gate in summary.gates if gate.name == "tool_execution_policy").passed
    restored = _roundtrip(tmp_path, result, summary)
    assert restored.retrieval_diagnostics[-1].status == "unavailable"


@pytest.mark.parametrize("delivery_tools", [{"/chat.postMessage": None}], indirect=True)
def test_actual_unknown_slack_delivery_reuses_failed_origin_without_resending(delivery_tools, tmp_path):
    _save, slack, sent_requests, _output = delivery_tools
    contract = _contract(slack="requested", slack_recipient={
        "state": "explicit", "selector": {"kind": "channel", "value": "C123"},
        "evidence_ids": ["request"],
    })

    def prepare(state):
        prepared = _state("Unconfirmed delivery", contract=contract)
        prepared["runtime"] = prepared["runtime"].model_copy(update={
            "pending_action": state["runtime"].pending_action,
        })
        return prepared

    manager = _manager(prepare=prepare, action=make_action_postprocess_node(None, slack, False))
    first = _runtime_payload(manager.run_agent_flow("Send to C123"))
    second = _runtime_payload(manager.run_agent_flow("Continue the same task"))
    original = first["debug"]["execution_evidence"]["events"]
    reused = second["debug"]["execution_evidence"]["events"]
    assert [event["phase"] for event in original] == ["started", "failed"]
    assert [event["phase"] for event in reused] == ["reused"]
    assert reused[0]["origin_invocation_id"] == original[0]["invocation_id"]

    result, summary = _evaluate([first, second], final_forbidden=["slack_notify"], slack_required=True)

    assert result.policy_assessment.status == "compliant"
    assert len([request for request in sent_requests if request["path"] == "/chat.postMessage"]) == 1
    assert result.tool_calls == []
    assert result.tool_call_count == 0
    assert result.slack_delivery_status == "unknown"
    assert result.actions[0].status == result.actions[0].slack.status == "unknown"
    assert result.actions[0].invocation_id == original[0]["invocation_id"]
    assert summary.metrics.slack_delivery_success_cases == 0
    assert next(gate for gate in summary.gates if gate.name == "slack_delivery_success_rate").gate_type == "audit"
    restored = _roundtrip(tmp_path, result, summary)
    assert restored.slack_delivery_status == "unknown"
    assert restored.actions[0].slack == result.actions[0].slack


def _payload(request_id, phases, *, tool="upload_search", status="complete"):
    events = []
    for phase in phases:
        event = {"sequence": len(events) + 1, "invocation_id": "origin" if phase != "reused" else "reuse",
                 "tool_name": tool, "phase": phase}
        if phase == "reused":
            event["origin_invocation_id"] = "origin"
        events.append(event)
    evidence = {"schema_version": 1, "request_id": request_id, "status": status, "events": events}
    payload = _final_payload(plain_response("Observed result"),
                             tool_calls=[tool] if "started" in phases else [],
                             debug_overrides={"execution_evidence": evidence})
    payload["trace"] = f"Request ID: {request_id}"
    return payload


@pytest.mark.parametrize("terminal", ["failed", "succeeded"])
@pytest.mark.parametrize("previous_turn", [False, True])
def test_completed_origin_is_reusable_in_event_order(terminal, previous_turn):
    payloads = ([_payload("setup", ["started", terminal]), _payload("final", ["reused"])] if previous_turn
                else [_payload("final", ["started", terminal, "reused"])])

    result, _ = _evaluate(payloads)

    assert result.policy_assessment.status == "compliant"
    assert result.tool_call_count == (0 if previous_turn else 1)


@pytest.mark.parametrize("previous_turn", [False, True])
def test_failed_reuse_preserves_the_original_forbidden_execution(previous_turn):
    payloads = ([_payload("setup", ["started", "failed"]), _payload("final", ["reused"])] if previous_turn
                else [_payload("final", ["started", "failed", "reused"])])

    result, summary = _evaluate(payloads, setup_forbidden=[["upload_search"]] if previous_turn else [],
                                final_forbidden=["upload_search"])

    assert result.policy_assessment.status == "violated"
    assert "forbidden_tool_execution" in result.gate_failures
    assert "tool_execution_reuse_unverifiable" not in result.gate_failures
    assert not next(gate for gate in summary.gates if gate.name == "tool_execution_policy").passed


@pytest.mark.parametrize("defect,expected_issue", [
    ("missing", "tool_execution_reuse_unverifiable"),
    ("tool_mismatch", "tool_execution_reuse_unverifiable"),
    ("start_only", "tool_execution_evidence_incomplete"),
    ("incomplete", "tool_execution_evidence_incomplete"),
    ("request_mismatch", "tool_execution_request_mismatch"),
    ("blocked", "tool_execution_reuse_unverifiable"),
    ("completion_after_reuse", "tool_execution_reuse_unverifiable"),
])
def test_unproven_origins_are_never_registered(defect, expected_issue):
    # A successful terminal is deliberate: malformed evidence must not become
    # trusted merely because an individual event describes success.
    first = _payload("setup", ["started", "succeeded"])
    final = _payload("final", ["reused"])
    evidence = first["debug"]["execution_evidence"]
    if defect == "missing":
        first = _payload("setup", [])
    elif defect == "tool_mismatch":
        first = _payload("setup", ["started", "succeeded"], tool="tavily_search")
    elif defect == "start_only":
        first = _payload("setup", ["started"], status="incomplete")
    elif defect == "incomplete":
        evidence["status"] = "incomplete"
    elif defect == "request_mismatch":
        evidence["request_id"] = "other-request"
    elif defect == "blocked":
        first = _payload("setup", ["blocked"])
    elif defect == "completion_after_reuse":
        final = _payload("final", ["started", "reused", "succeeded"])
        first = _payload("setup", [])

    result, _ = _evaluate([first, final])

    assert result.policy_assessment.status == "indeterminate"
    assert expected_issue in result.gate_failures
    assert "tool_execution_reuse_unverifiable" in result.gate_failures
    assert result.release_pass is False


@pytest.mark.parametrize("selected_has_origin", [False, True])
def test_conflicting_debug_history_cannot_supply_a_trusted_origin(selected_has_origin):
    first = _payload("setup", ["started", "succeeded"] if selected_has_origin else [])
    final = _payload("final", ["reused"])
    parsed = parse_agent_response(first)
    # The retained scenario evidence and debug envelope disagree. The debug
    # copy may expose a violation, but cannot prove the selected history's origin.
    alternate = _payload("setup", [] if selected_has_origin else ["started", "succeeded"])["debug"]["execution_evidence"]
    turn = ScenarioTurnResult(
        query="Prepare", request_payload={}, request_id=parsed.request_id,
        response=parsed.response, execution_evidence=parsed.execution_evidence,
        tool_calls=parsed.tool_calls, debug={**parsed.debug, "execution_evidence": alternate},
    )
    result, _ = _evaluate([first, final], retained_turns=[turn])

    assert result.policy_assessment.status == "indeterminate"
    assert "tool_execution_observation_conflict" in result.gate_failures
    assert "tool_execution_reuse_unverifiable" in result.gate_failures
