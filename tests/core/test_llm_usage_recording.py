from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src.app.agent_manager import AgentFlowManager
from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import GraphState, ResponseState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.llm_errors import LLMCallError
from src.infra.settings import AppSettings
from src.runtime.agent_runtime.llm_usage import capture_llm_usage, current_llm_calls, record_llm_call
from src.runtime.nodes.planner.node import make_planner_node
from src.runtime.nodes.session import make_summarize_node


def test_capture_distinguishes_unavailable_evidence_from_no_model_invocation():
    assert current_llm_calls() is None
    with capture_llm_usage() as recorder:
        assert recorder.snapshot() == []
        with record_llm_call(stage="synthesis", attempt=1, path="structured") as call:
            call.complete(AIMessage(content="", usage_metadata={
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            }))
        assert recorder.snapshot()[0].usage.total_tokens == 0
        with capture_llm_usage() as nested:
            assert nested.snapshot() == []
        assert len(current_llm_calls()) == 1
    assert current_llm_calls() is None


class UnavailableModel:
    def invoke(self, _messages):
        raise TimeoutError("provider timeout")


def test_failed_planner_and_summary_remain_observable_attempts():
    state = build_graph_state_input(user_input="current", messages=[
        HumanMessage(content="first"), AIMessage(content="answer"),
        HumanMessage(content="recent"), AIMessage(content="recent answer"),
        HumanMessage(content="current"),
    ])
    with capture_llm_usage() as recorder:
        summary = make_summarize_node(UnavailableModel(), False, policy=ConversationMemoryPolicy(
            high_water_turns=3, low_water_turns=2,
        ))(state)
        with pytest.raises(LLMCallError) as caught:
            make_planner_node(UnavailableModel(), False)(state)

    assert summary["debug"].memory_compactions[0]["summary_fallback"]
    assert caught.value.problem.code == "provider_unavailable"
    calls = recorder.snapshot()
    assert [call.stage for call in calls] == ["summarize", "planner", "planner", "planner"]
    assert all(call.usage.input_tokens is None and call.usage.output_tokens is None for call in calls)


@pytest.mark.parametrize("failure_phase", ["graph", "collector", "assembly"])
def test_manager_retains_completed_usage_after_later_pipeline_failure(monkeypatch, failure_phase):
    manager = AgentFlowManager(AppSettings(_env_file=None, openai_api_key="test", tavily_api_key="test"))

    def synthesize(state):
        with record_llm_call(stage="synthesis", attempt=1, path="structured") as call:
            call.complete(AIMessage(content="answer", response_metadata={"model_name": "test-model"},
                usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}))
        if failure_phase == "graph":
            raise RuntimeError("graph failure after model response")
        return {
            "messages": [HumanMessage(content=state["runtime"].user_input), AIMessage(content="answer")],
            "response": ResponseState(result=finalize_answer(text_document("answer"), [])),
        }

    graph = StateGraph(GraphState)
    graph.add_node("synthesis", synthesize)
    graph.add_edge(START, "synthesis")
    graph.add_edge("synthesis", END)
    manager.graph = graph.compile()

    if failure_phase != "graph":
        owner, method = ((manager._debug_collector, "build") if failure_phase == "collector"
                         else (manager._response_assembler, "assemble"))
        original = getattr(owner, method)

        def fail_after_result(**kwargs):
            original(**kwargs)
            raise RuntimeError(f"{failure_phase} failure after model response")

        monkeypatch.setattr(owner, method, fail_after_result)

    try:
        result = manager.run_agent_flow("question")
        assert result["debug"]["observability_status"] == "failed"
        calls = result["debug"]["llm_calls"]
        assert len(calls) == 1
        assert calls[0]["model_name"] == "test-model"
        assert calls[0]["usage"]["input_tokens"] == 100
        assert calls[0]["usage"]["output_tokens"] == 20
        assert "token_usage" not in result["debug"]
        assert "model_usage_status" not in result["debug"]
    finally:
        manager.close()
