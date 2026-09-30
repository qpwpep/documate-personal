from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src.app.agent_manager import AgentFlowManager
from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import GraphState, ResponseState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.conversation_memory import ConversationMemoryPolicy
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
        planner = make_planner_node(UnavailableModel(), False)(state)

    assert summary["debug"].memory_compactions[0]["summary_fallback"]
    assert planner["planner"].status == "fallback_no_routes"
    calls = recorder.snapshot()
    assert [call.stage for call in calls] == ["summarize", "planner"]
    assert all(call.usage.input_tokens is None and call.usage.output_tokens is None for call in calls)

