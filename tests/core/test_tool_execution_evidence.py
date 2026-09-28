"""Execution evidence survives graph failures and distinguishes nonexecution."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from src.app.agent_manager import AgentFlowManager
from src.app.web.agent_request_support import normalize_debug_info
from src.core.contracts import GraphState, PlannerState
from src.core.planner_schema import PlannerOutput, RetrievalRequirement, RetrievalTask
from src.infra.settings import AppSettings
from src.infra.tools.local_rag import build_upload_search_tool
from src.infra.tools.docs_search import build_docs_search_tool
from src.infra.tools.save_text import build_save_text_tool
from src.infra.tools.slack_notify import build_slack_notify_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from tests.core.test_actions_nodes import _contract, _state


def _manager(*, prepare, action, fail_after=False):
    def prepared(state):
        updates = prepare(state)
        if "response" in updates:
            updates["response"] = updates["response"].model_copy(update={"body_kind": "compose"})
        return updates
    graph = StateGraph(GraphState)
    graph.add_node("prepare", prepared)
    graph.add_node("execute", action)
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "execute")
    if fail_after:
        def fail(_state):
            raise ValueError("later graph failure")
        graph.add_node("fail", fail)
        graph.add_edge("execute", "fail")
        graph.add_edge("fail", END)
    else:
        graph.add_edge("execute", END)
    manager = AgentFlowManager.__new__(AgentFlowManager)
    manager.settings = AppSettings(openai_api_key="test-key")
    manager.graph = graph.compile()
    return manager


@pytest.mark.parametrize("fail_after", [False, True])
def test_saved_execution_survives_success_and_later_graph_failure(tmp_path, monkeypatch, fail_after):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    manager = _manager(prepare=lambda _: _state("saved body"),
                       action=make_action_postprocess_node(build_save_text_tool(), None, False),
                       fail_after=fail_after)

    response = manager.run_agent_flow("save the answer")
    debug = normalize_debug_info(response["debug"], 1).model_dump(mode="json")

    assert list(tmp_path.glob("*.txt"))
    evidence = debug["execution_evidence"]
    assert evidence["status"] == "complete"
    assert [(event["tool_name"], event["phase"]) for event in evidence["events"]] == [
        ("save_text", "started"), ("save_text", "succeeded"),
    ]
    assert evidence["events"][0]["invocation_id"] == evidence["events"][1]["invocation_id"]
    if not fail_after:
        assert response["response"]["actions"][0]["invocation_id"] == evidence["events"][0]["invocation_id"]
    assert debug["tool_calls"] == ["save_text"]
    assert debug["observability_status"] == ("failed" if fail_after else "ok")


def test_forbidden_intent_is_blocked_before_adapter_entry(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    manager = _manager(prepare=lambda _: _state("do not save", contract=_contract(save="forbidden")),
                       action=make_action_postprocess_node(build_save_text_tool(), None, False))

    debug = manager.run_agent_flow("do not save")["debug"]

    assert list(tmp_path.iterdir()) == []
    assert debug["execution_evidence"]["status"] == "complete"
    assert [(event["tool_name"], event["phase"]) for event in debug["execution_evidence"]["events"]] == [
        ("save_text", "blocked"),
    ]
    assert debug["tool_calls"] == []


def test_parallel_adapters_returning_failure_still_record_execution():
    def prepare(_):
        state = _state("unavailable uploads", contract=_contract())
        state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[
            RetrievalTask(route="upload", query=query, k=1) for query in ("alpha", "beta")
        ]))
        return state
    manager = _manager(prepare=prepare,
                       action=make_retrieve_dispatch_node(None, build_upload_search_tool(), False))

    debug = manager.run_agent_flow("search uploads")["debug"]

    events = debug["execution_evidence"]["events"]
    assert debug["execution_evidence"]["status"] == "complete"
    assert [item["sequence"] for item in events] == list(range(1, 5))
    assert len([item for item in events if item["phase"] == "started"]) == 2
    assert len([item for item in events if item["phase"] == "failed"]) == 2
    assert {item["tool_name"] for item in events} == {"upload_search"}


def test_planned_message_does_not_invent_an_execution():
    def prepare(_):
        state = _state("plan only", contract=_contract())
        state["messages"] = [HumanMessage(content="plan only"), AIMessage(content="", tool_calls=[
            {"name": "save_text", "args": {}, "id": "planned-call"},
        ])]
        return state
    manager = _manager(prepare=prepare, action=lambda _: {})

    debug = manager.run_agent_flow("plan only")["debug"]

    assert debug["execution_evidence"]["status"] == "complete"
    assert debug["execution_evidence"]["events"] == []
    assert debug["tool_calls"] == []


def test_adapter_failure_after_partial_write_retains_start_and_receipt_identity(tmp_path, monkeypatch):
    import src.infra.saved_artifacts as storage

    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    write = storage.os.write
    def partial_write(descriptor, content):
        write(descriptor, content[:3])
        raise OSError("disk full after partial write")
    monkeypatch.setattr(storage.os, "write", partial_write)
    manager = _manager(prepare=lambda _: _state("saved body"),
                       action=make_action_postprocess_node(build_save_text_tool(), None, False))

    response = manager.run_agent_flow("save the answer")

    events = response["debug"]["execution_evidence"]["events"]
    assert [item["phase"] for item in events] == ["started", "failed"]
    assert response["response"]["actions"][0]["status"] == "error"
    assert response["response"]["actions"][0]["invocation_id"] == events[0]["invocation_id"]


def test_authentication_failure_inside_slack_adapter_is_an_execution():
    contract = _contract(slack="requested", slack_recipient={
        "state": "explicit", "selector": {"kind": "channel", "value": "C123"}, "evidence_ids": ["request"],
    })
    manager = _manager(prepare=lambda _: _state("send body", contract=contract),
                       action=make_action_postprocess_node(None, build_slack_notify_tool(None), False))

    response = manager.run_agent_flow("send to C123")

    evidence = response["debug"]["execution_evidence"]
    assert evidence["status"] == "complete"
    assert [event["phase"] for event in evidence["events"]] == ["started", "failed"]
    receipt = response["response"]["actions"][0]
    assert receipt["slack"]["status"] == "not_sent"
    assert receipt["invocation_id"] == receipt["slack"]["invocation_id"] == evidence["events"][0]["invocation_id"]


def test_docs_adapter_entry_is_recorded_even_without_a_provider_request():
    def prepare(_):
        state = _state("unsupported library", contract=_contract())
        state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[
            RetrievalTask(route="docs", query="unsupported docs", k=1,
                          requirement=RetrievalRequirement(library="unsupported-test-library")),
        ]))
        return state
    manager = _manager(prepare=prepare, action=make_retrieve_dispatch_node(
        build_docs_search_tool(AppSettings(openai_api_key="test-key")), None, False,
    ))

    debug = manager.run_agent_flow("search unsupported docs")["debug"]

    events = debug["execution_evidence"]["events"]
    assert [(event["tool_name"], event["phase"]) for event in events] == [
        ("tavily_search", "started"), ("tavily_search", "succeeded"),
    ]
    assert debug["retrieval_diagnostics"][0]["status"] == "no_result"
    assert debug["retrieval_diagnostics"][0]["invocation_id"] == events[0]["invocation_id"]


def test_reused_retrieval_has_origin_without_a_second_execution():
    def prepare(_):
        state = _state("unavailable uploads", contract=_contract())
        state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[
            RetrievalTask(route="upload", query="alpha", k=1),
        ]))
        return state
    retrieve = make_retrieve_dispatch_node(None, build_upload_search_tool(), False)
    def retrieve_twice(state):
        first = retrieve(state)
        return retrieve({**state, **first})
    manager = _manager(prepare=prepare, action=retrieve_twice)

    debug = manager.run_agent_flow("search once")["debug"]

    events = debug["execution_evidence"]["events"]
    assert [event["phase"] for event in events] == ["started", "failed", "reused"]
    assert events[-1]["origin_invocation_id"] == events[0]["invocation_id"]
    assert debug["retrieval_diagnostics"][-1]["invocation_id"] == events[0]["invocation_id"]
    assert debug["tool_calls"] == ["upload_search"]


def test_malformed_envelope_keeps_known_started_event_at_http_boundary():
    from src.core.contracts.debug import DebugPayload
    raw = DebugPayload().model_dump(mode="json")
    raw["execution_evidence"] = {
        "schema_version": 1, "request_id": "r1", "status": "complete", "events": [
            {"sequence": 1, "invocation_id": "i1", "tool_name": "save_text", "phase": "started"},
        ],
    }

    debug = normalize_debug_info(raw, 1).model_dump(mode="json")

    assert debug["execution_evidence"] == raw["execution_evidence"]
    assert debug["observability_status"] == "failed"
    assert "execution_evidence" in debug["missing_required_debug_fields"]


def test_unfinished_execution_is_incomplete_and_missing_capture_stays_unavailable():
    from src.core.contracts.debug import DebugPayload
    from src.runtime.agent_runtime.tool_execution import ToolExecutionRecorder
    recorder = ToolExecutionRecorder("r1")
    recorder.record("save_text", "started")

    evidence = recorder.snapshot().model_dump(mode="json")

    assert evidence["status"] == "incomplete"
    assert [event["phase"] for event in evidence["events"]] == ["started"]
    assert DebugPayload().execution_evidence is None


def test_saved_receipt_reuse_preserves_origin_across_requests(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    contract = _contract(save="requested", slack="requested")
    def prepare(state):
        prepared = _state("same saved body", contract=contract)
        prepared["runtime"] = prepared["runtime"].model_copy(update={
            "pending_action": state["runtime"].pending_action,
        })
        return prepared
    manager = _manager(prepare=prepare,
                       action=make_action_postprocess_node(build_save_text_tool(), build_slack_notify_tool(None), False))

    first = manager.run_agent_flow("save and send")
    second = manager.run_agent_flow("finish the same request")

    receipt = next(action for action in first["response"]["actions"] if action["kind"] == "save_text")
    reused_receipt = next(action for action in second["response"]["actions"] if action["kind"] == "save_text")
    evidence = second["debug"]["execution_evidence"]
    assert evidence["status"] == "complete"
    reused = next(event for event in evidence["events"] if event["tool_name"] == "save_text")
    assert reused["phase"] == "reused"
    assert reused["origin_invocation_id"] == receipt["invocation_id"] == reused_receipt["invocation_id"]
    assert second["debug"]["tool_calls"] == []
    assert len(list(tmp_path.glob("*.txt"))) == 1


@pytest.mark.parametrize("missing_field", ["schema_version", "events"])
def test_complete_evidence_requires_explicit_version_and_event_inventory(missing_field):
    from pydantic import ValidationError
    from src.core.contracts.tool_execution import ToolExecutionEvidence
    raw = {"schema_version": 1, "request_id": "r1", "status": "complete", "events": []}
    raw.pop(missing_field)

    with pytest.raises(ValidationError):
        ToolExecutionEvidence.model_validate(raw)


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_execution_schema_version_is_an_explicit_integer(version):
    from pydantic import ValidationError
    from src.core.contracts.tool_execution import ToolExecutionEvidence

    with pytest.raises(ValidationError):
        ToolExecutionEvidence.model_validate({"schema_version": version, "request_id": "r1", "status": "complete", "events": []})


@pytest.mark.parametrize("tool_name", [" ", "save_text ", "\tupload_search"])
def test_execution_tool_names_cannot_hide_in_whitespace(tool_name):
    from pydantic import ValidationError
    from src.core.contracts.tool_execution import ToolExecutionEvent

    with pytest.raises(ValidationError):
        ToolExecutionEvent(sequence=1, invocation_id="i1", tool_name=tool_name, phase="blocked")


def test_http_normalization_does_not_hide_explicit_zero_call_count():
    from src.core.contracts.debug import DebugPayload
    raw = DebugPayload().model_dump(mode="json")
    raw.update(tool_calls=["save_text"], tool_call_count=0, execution_evidence={
        "schema_version": 1, "request_id": "r1", "status": "complete", "events": [
            {"sequence": 1, "invocation_id": "i1", "tool_name": "save_text", "phase": "started"},
            {"sequence": 2, "invocation_id": "i1", "tool_name": "save_text", "phase": "succeeded"},
        ],
    })

    debug = normalize_debug_info(raw, 1).model_dump(mode="json")

    assert debug["tool_call_count"] == 0
    assert debug["execution_evidence"]["events"][0]["phase"] == "started"
    assert "execution_evidence" in debug["missing_required_debug_fields"]
    assert debug["observability_status"] == "failed"
