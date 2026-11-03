from __future__ import annotations

import json
import socket

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.core.contracts import GraphState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.conversation_memory import (
    build_durable_conversation_memory,
    estimate_text_tokens,
)
from src.infra import llm as llm_module
from src.infra.llm_boundary import capture_llm_diagnostics
from src.infra.settings import APP_ENV_SPEC_BY_NAME, AppSettings
from src.runtime.agent_runtime.llm_usage import capture_llm_usage
from src.runtime.agent_runtime.session_context import SessionContext
from src.runtime.nodes.planner.prompt_builder import build_planner_messages
from src.runtime.nodes.session import make_summarize_node


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in APP_ENV_SPEC_BY_NAME:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    def reject_network(*_args, **_kwargs):
        pytest.fail("Summary budget tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


def _settings(**overrides) -> AppSettings:
    return AppSettings(
        _env_file=None,
        openai_api_key="test-summary-key",
        tavily_api_key="test-tavily-key",
        chat_model="gpt-5.6-luna",
        planner_model="gpt-5.6-luna",
        summary_model="gpt-5.6-luna",
        verbose=False,
        **overrides,
    )


@pytest.fixture
def summary_api(monkeypatch):
    requests = []
    outcome = {"content": "", "status_code": 200}

    def respond(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("POST", "/v1/chat/completions")
        requests.append({**json.loads(request.content), "timeout": request.extensions["timeout"]})
        statuses = outcome.get("status_codes", [outcome["status_code"]])
        status_code = statuses[min(len(requests) - 1, len(statuses) - 1)]
        if status_code != 200:
            return httpx.Response(
                status_code,
                headers={"retry-after": outcome.get("retry_after", "0")},
                json={"error": {"message": "Summary request rejected", "type": "invalid_request_error"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "summary-test-response",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5.6-luna",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": outcome["content"]},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        def create_chat_model(**kwargs):
            return ChatOpenAI(
                **kwargs,
                http_client=client,
                base_url="https://summary-tests.invalid/v1",
            )

        monkeypatch.setattr(llm_module, "ChatOpenAI", create_chat_model)
        yield outcome, requests


@pytest.mark.parametrize(
    ("generation_budget", "storage_budget", "storage_bytes", "content", "status_code", "expected_status"),
    [
        pytest.param(1024, 256, 4096, 'REPLACEMENT_FACT: Python 3.12 한글😀"\n\x00' * 100, 200, "ok", id="large-summary"),
        pytest.param(2048, 256, 4096, "", 400, "degraded", id="summary-error"),
        pytest.param(1024, 256, 4096, " \n ", 200, "degraded", id="blank-summary"),
        pytest.param(1024, 256, 512, "REPLACEMENT_FACT: keep Python 3.12 " * 100, 200, "ok", id="byte-limit-summary"),
    ],
)
def test_generated_or_fallback_summary_stays_bounded_and_reaches_next_prompt(
    summary_api, generation_budget, storage_budget, storage_bytes, content, status_code, expected_status,
):
    """Generation budgets reach the API while stored and reused memory obeys its own bounds."""
    outcome, requests = summary_api
    outcome.update(content=content, status_code=status_code)
    settings = _settings(
        summary_max_tokens=generation_budget,
        memory_summary_max_tokens=storage_budget,
        memory_summary_max_bytes=storage_bytes,
        memory_high_water_turns=3,
        memory_low_water_turns=2,
    )
    policy = settings.conversation_memory_policy()
    registry = llm_module.build_llm_registry(settings)
    builder = StateGraph(GraphState)
    builder.add_node("summarize", make_summarize_node(registry.llm_summarizer, False, policy=policy))
    builder.set_entry_point("summarize")
    builder.add_edge("summarize", END)
    state = builder.compile().invoke(build_graph_state_input(
        user_input="current request",
        memory_summary="OLDER_FACT: preserve the prior constraint " * 30,
        messages=[
            HumanMessage(content='EVICTED_FACT: Python 3.12 "\n\x00' * 40),
            AIMessage(content="old answer " * 20),
            ToolMessage(content="SECRET_TOOL_PAYLOAD", name="search", tool_call_id="old-search"),
            HumanMessage(content="recent request"),
            AIMessage(content="recent answer"),
            HumanMessage(content="current request"),
        ],
    ))
    memory = build_durable_conversation_memory(
        [*state["messages"], AIMessage(content="current answer")],
        memory_summary=state["runtime"].memory_summary,
        policy=policy,
    )
    session = SessionContext()
    session.commit_conversation_memory(messages=memory.messages, memory_summary=memory.memory_summary)
    snapshot = session.snapshot_conversation_memory()
    summary = snapshot.memory_summary
    assert summary
    assert state["runtime"].memory_summary == summary
    assert estimate_text_tokens(summary) <= storage_budget
    assert len(summary.encode("utf-8")) <= policy.summary_max_bytes
    json_content_bytes = len(json.dumps(summary, ensure_ascii=False).encode("utf-8")) - 2
    assert json_content_bytes <= min(policy.summary_max_bytes, storage_budget * 3)
    assert memory.usage.serialized_bytes < policy.high_water_bytes
    assert memory.usage.serialized_bytes <= policy.hard_max_bytes
    assert [(type(message), message.content) for message in snapshot.messages] == [
        (HumanMessage, "recent request"),
        (AIMessage, "recent answer"),
        (HumanMessage, "current request"),
        (AIMessage, "current answer"),
    ]
    assert state["debug"].observability_status == expected_status
    diagnostic = state["debug"].memory_compactions[-1]
    assert diagnostic["summary_max_bytes"] == min(policy.summary_max_bytes, storage_budget * 3)
    assert diagnostic["summary_stored_bytes"] == json_content_bytes
    expected_generated_bytes = len(json.dumps(content, ensure_ascii=False).encode("utf-8")) - 2
    assert diagnostic["summary_generated_bytes"] == (expected_generated_bytes if status_code == 200 else None)
    if expected_status == "ok":
        assert summary.startswith("REPLACEMENT_FACT:")
        assert "OLDER_FACT:" not in summary
    else:
        assert "OLDER_FACT:" in summary

    next_prompt = build_planner_messages(build_graph_state_input(
        user_input="follow-up request",
        messages=[*snapshot.messages, HumanMessage(content="follow-up request")],
        memory_summary=summary,
    ), max_turns=policy.low_water_turns)
    prompt_memory = [
        json.loads(message.content)
        for message in next_prompt
        if isinstance(message, AIMessage) and str(message.content).startswith("{")
    ]
    assert prompt_memory == [{"kind": "untrusted_conversation_memory", "summary": summary}]
    context = json.loads(next(message.content for message in next_prompt
                              if message.name == "request_context").split("\n", 1)[1])
    assert context["user_turn_ledger"][-1]["text"] == "follow-up request"
    assert next_prompt[-1] == HumanMessage(content="follow-up request")
    assert sum(str(message.content).count("follow-up request") for message in next_prompt) == 2
    assert [request["max_completion_tokens"] for request in requests] == [generation_budget]
    transcript = requests[0]["messages"][-1]["content"]
    assert "OLDER_FACT:" in transcript
    assert "EVICTED_FACT:" in transcript
    assert "SECRET_TOOL_PAYLOAD" not in transcript


@pytest.mark.parametrize(
    ("statuses", "retry_after", "expected_statuses", "expected_recoveries"),
    [
        ([503, 200], "0", ["failed", "completed"], ["retry"]),
        ([401], "0", ["failed"], ["stop"]),
        ([503], "0", ["failed", "failed", "failed"], ["retry", "retry", "stop"]),
        ([429], "120", ["failed"], ["stop"]),
    ],
)
def test_summary_records_every_attempt_and_bounds_transient_retries(
    summary_api, statuses, retry_after, expected_statuses, expected_recoveries,
):
    outcome, requests = summary_api
    outcome.update(content="Python 3.12 remains required.", status_codes=statuses, retry_after=retry_after)
    settings = _settings(memory_high_water_turns=3, memory_low_water_turns=2)
    policy = settings.conversation_memory_policy()
    registry = llm_module.build_llm_registry(settings)
    summarize = make_summarize_node(registry.llm_summarizer, False, policy=policy)
    with capture_llm_usage() as usage, capture_llm_diagnostics() as diagnostics:
        state = summarize(build_graph_state_input(
            user_input="current request",
            memory_summary="Keep source exclusions.",
            messages=[
                HumanMessage(content="Python 3.12 remains required."),
                AIMessage(content="Understood."),
                HumanMessage(content="recent request"),
                AIMessage(content="recent answer"),
                HumanMessage(content="current request"),
            ],
        ))
    calls = usage.snapshot()
    assert len(requests) == len(expected_statuses)
    assert all(0 < request["timeout"]["read"] <= 45 for request in requests)
    assert [call.status for call in calls] == expected_statuses
    assert [call.attempt for call in calls] == list(range(1, len(requests) + 1))
    assert all(call.stage == "summarize" for call in calls)
    assert all(call.model_name == "gpt-5.6-luna" for call in calls)
    assert all(call.endpoint == "chat_completions" for call in calls)
    assert all(call.latency_ms is not None for call in calls)
    assert [item.recovery for item in diagnostics] == expected_recoveries
    for call in calls:
        if call.status == "failed":
            assert call.usage.input_tokens is None
            assert call.usage.output_tokens is None
            assert call.failure_type is not None
        else:
            assert (call.usage.input_tokens, call.usage.output_tokens) == (100, 20)
    assert "Python 3.12" in state["runtime"].memory_summary
    assert state["debug"].memory_compactions[-1]["summary_fallback"] is (expected_statuses[-1] == "failed")
