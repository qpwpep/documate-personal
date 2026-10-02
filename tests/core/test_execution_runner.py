from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.app.agent_manager import AgentFlowManager
from src.core.answer_schema import finalize_answer, text_document
from src.core.conversation_memory import DEFAULT_QUERY_MAX_CHARS
from src.core.contracts import DebugState, GraphState, ResponseState
from src.core.latency import make_stage_latency_event
from src.infra.settings import AppSettings
from src.runtime.agent_runtime import ExecutionRunner, GraphInvocationError, SessionContext
from src.runtime.graph_builder import _instrument_stage_node
from src.runtime.progress import ProgressEmitter


class _ExecutionState(TypedDict, total=False):
    route_decisions: Annotated[list[dict[str, Any]], add]


def _decision(sequence: int = 1) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "source": "planner",
        "target": "synthesize",
        "reason": "retrieval_not_required",
    }


def _runner(graph: Any) -> ExecutionRunner:
    return ExecutionRunner(
        graph=graph,
        session=SessionContext(),
    )


def test_failure_preserves_the_decision_committed_before_the_failing_node() -> None:
    visited: list[str] = []

    def decide(_state: _ExecutionState):
        visited.append("planner")
        return Command(update={"route_decisions": [_decision()]}, goto="synthesize")

    def fail(_state: _ExecutionState):
        visited.append("synthesize")
        raise RuntimeError("synthesis failed")

    builder = StateGraph(_ExecutionState)
    builder.add_node("planner", decide)
    builder.add_node("synthesize", fail)
    builder.add_edge(START, "planner")
    builder.add_edge("synthesize", END)

    with pytest.raises(GraphInvocationError, match="synthesis failed") as caught:
        _runner(builder.compile()).invoke_graph({"route_decisions": []})

    assert visited == ["planner", "synthesize"]
    assert caught.value.last_state["route_decisions"] == [_decision()]
    assert caught.value.graph_total_ms >= 0


def test_success_returns_last_state_without_accumulating_snapshot_histories() -> None:
    repeated = {"sequence": 2, "source": "post_synthesis_validation", "target": "synthesize", "reason": "missing_content"}
    finished = {"sequence": 3, "source": "post_synthesis_validation", "target": "action_postprocess", "reason": "validation_passed"}

    def validate(state: _ExecutionState):
        sequence = len(state["route_decisions"]) + 1
        decision = repeated if sequence == 2 else finished
        return Command(update={"route_decisions": [decision]}, goto=decision["target"])

    builder = StateGraph(_ExecutionState)
    builder.add_node("planner", lambda _state: Command(update={"route_decisions": [_decision()]}, goto="synthesize"))
    builder.add_node("synthesize", lambda _state: {})
    builder.add_node("post_synthesis_validation", validate)
    builder.add_node("action_postprocess", lambda _state: {})
    builder.add_edge(START, "planner")
    builder.add_edge("synthesize", "post_synthesis_validation")
    builder.add_edge("action_postprocess", END)

    result, graph_total_ms = _runner(builder.compile()).invoke_graph({"route_decisions": []})

    assert result["route_decisions"] == [_decision(), repeated, finished]
    assert graph_total_ms >= 0


def test_failure_before_first_decision_keeps_an_empty_committed_history() -> None:
    def fail(_state: _ExecutionState):
        raise RuntimeError("planner failed")

    builder = StateGraph(_ExecutionState)
    builder.add_node("planner", fail)
    builder.add_edge(START, "planner")

    with pytest.raises(GraphInvocationError, match="planner failed") as caught:
        _runner(builder.compile()).invoke_graph({"route_decisions": []})

    assert caught.value.last_state["route_decisions"] == []


_COMPACTION = {"reason": "high_watermark", "removed_messages": 4, "summary_fallback": False}
_COMPLETED_LATENCY = make_stage_latency_event(stage="synthesis", attempt=1, latency_ms=9)


def _manager(graph: Any) -> AgentFlowManager:
    manager = AgentFlowManager.__new__(AgentFlowManager)
    manager.settings = AppSettings(_env_file=None, openai_api_key="test-key", tavily_api_key="test-key")
    manager.graph = graph
    manager.messages = [HumanMessage(content="previous request"), AIMessage(content="previous answer")]
    manager.memory_summary = "previous summary"
    return manager


def _manager_graph(outcome: str):
    def decide(_state: GraphState):
        return Command(update={
            "route_decisions": [_decision()],
            "debug": DebugState(memory_compactions=[_COMPACTION], latency_trace=[_COMPLETED_LATENCY]),
            "response": ResponseState(synthesis_attempt=1),
        }, goto="synthesize")

    def finish(state: GraphState):
        if outcome == "graph_failure":
            raise RuntimeError("synthesis failed")
        response = ResponseState(result=finalize_answer(text_document("new answer"), []), kind="answer")
        updates: dict[str, Any] = {
            "messages": [HumanMessage(content="new request"), AIMessage(content="new answer")],
            "response": response,
        }
        if outcome == "collector_failure":
            debug = state["debug"].model_dump(mode="json")
            debug["tool_call_count"] = "invalid count"
            updates["debug"] = debug
        elif outcome == "assembly_failure":
            response.result.content.blocks[0].content[0].text = "changed after validation"
        elif outcome == "runtime_failure":
            updates["runtime"] = {"previous_response": {"content": text_document("unchecked runtime body").model_dump(mode="json")}}
        return updates

    builder = StateGraph(GraphState)
    builder.add_node("planner", decide)
    builder.add_node("synthesize", _instrument_stage_node("synthesis", finish) if outcome == "graph_failure" else finish)
    builder.add_edge(START, "planner")
    builder.add_edge("synthesize", END)
    return builder.compile()


@pytest.mark.parametrize("outcome", ["graph_failure", "collector_failure", "assembly_failure", "runtime_failure"])
def test_manager_preserves_committed_diagnostics_and_session_on_failure(outcome: str) -> None:
    manager = _manager(_manager_graph(outcome))
    before = manager._ensure_session().snapshot_conversation_memory()
    progress: list[tuple[str, dict[str, Any]]] = []
    emitter = ProgressEmitter(publish=lambda event, data: progress.append((event, data)), request_id="request", session_id="session")

    result = manager.run_agent_flow("new request", progress_emitter=emitter)

    assert result["debug"]["observability_status"] == "failed"
    assert result["debug"]["route_decisions"] == [_decision()]
    assert result["debug"]["memory_compactions"] == [_COMPACTION]
    assert result["debug"]["latency_breakdown"]["graph_total_ms"] is not None
    assert result["debug"]["latency_breakdown"]["stage_attempts"][0] == {
        "stage": "synthesis", "attempt": 1, "latency_ms": 9, "status": None,
    }
    assert manager._ensure_session().snapshot_conversation_memory() == before
    assert not [event for event, _data in progress if event == "error"]
    assert result["status"] == "failed"
    assert result["response"] is None
    assert result["problem"]["code"] == "internal_error"
    if outcome == "graph_failure":
        attempts = result["debug"]["latency_breakdown"]["stage_attempts"]
        assert len(attempts) == 2
        assert attempts[-1]["stage"] == "synthesis"
        assert attempts[-1]["attempt"] == 2
        assert attempts[-1]["status"] == "error"


@pytest.mark.parametrize("query", ["", "exit"])
def test_manager_has_no_routing_decisions_before_graph_execution(query: str) -> None:
    result = _manager(_manager_graph("graph_failure")).run_agent_flow(query)

    assert result["debug"]["route_decisions"] == []
    assert result["debug"]["memory_compactions"] == []


@pytest.mark.parametrize("query", ["", "   ", "x" * (DEFAULT_QUERY_MAX_CHARS + 1)], ids=["empty", "blank", "too-long"])
def test_invalid_user_input_is_a_clarification_without_a_technical_failure(query: str) -> None:
    manager = _manager(_manager_graph("graph_failure"))
    before = manager._ensure_session().snapshot_conversation_memory()
    result = manager.run_agent_flow(query)
    assert result["status"] == "needs_input"
    assert result["problem"] is None
    assert result["response"] is None
    assert result["message"]
    assert result["missing_slots"] == ["query"]
    assert result["debug"]["error_codes"] == []
    assert result["debug"]["tool_calls"] == []
    assert manager._ensure_session().snapshot_conversation_memory() == before
