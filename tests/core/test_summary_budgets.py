from __future__ import annotations

import json
import socket

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from src.core.contracts import GraphState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.conversation_memory import (
    build_durable_conversation_memory,
    estimate_text_tokens,
)
from src.infra import llm as llm_module
from src.infra.settings import APP_ENV_SPEC_BY_NAME, AppSettings
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
        requests.append(json.loads(request.content))
        if outcome["status_code"] != 200:
            return httpx.Response(
                outcome["status_code"],
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
    ("generation_budget", "storage_budget", "content", "status_code", "expected_status"),
    [
        pytest.param(1024, 256, 'REPLACEMENT_FACT: Python 3.12 한글😀"\n\x00' * 100, 200, "ok", id="large-summary"),
        pytest.param(2048, 256, 'REPLACEMENT_FACT: Python 3.12 한글😀"\n\x00' * 100, 200, "ok", id="larger-generation-budget"),
        pytest.param(1024, 128, 'REPLACEMENT_FACT: Python 3.12 한글😀"\n\x00' * 100, 200, "ok", id="smaller-storage-budget"),
        pytest.param(1024, 256, "   ", 200, "degraded", id="blank-summary"),
        pytest.param(2048, 256, "", 400, "degraded", id="summary-error"),
    ],
)
def test_generated_or_fallback_summary_stays_bounded_and_reaches_next_prompt(
    summary_api, generation_budget, storage_budget, content, status_code, expected_status,
):
    """Generation budgets reach the API while stored and reused memory obeys its own bounds."""
    outcome, requests = summary_api
    outcome.update(content=content, status_code=status_code)
    settings = _settings(
        summary_max_tokens=generation_budget,
        memory_summary_max_tokens=storage_budget,
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
    assert next_prompt[-1] == HumanMessage(content="follow-up request")
    assert [request["max_completion_tokens"] for request in requests] == [generation_budget]


def test_generation_budget_can_exceed_memory_target_without_changing_memory_policy():
    """The API generation budget can grow beyond the storage target without enlarging stored memory."""
    default_policy = _settings().conversation_memory_policy()
    generation_budget = default_policy.low_water_tokens + 1
    settings = _settings(summary_max_tokens=generation_budget)

    assert settings.summary_max_tokens == generation_budget
    assert settings.conversation_memory_policy() == default_policy


@pytest.mark.parametrize("generation_budget", [0, -1])
def test_generation_budget_rejects_nonpositive_values(generation_budget):
    """The API output allowance must be positive even though it is independent of storage limits."""
    with pytest.raises(ValidationError):
        _settings(summary_max_tokens=generation_budget)


@pytest.mark.parametrize("dimension", ["tokens", "bytes"])
def test_stored_summary_reserve_must_still_fit_below_the_memory_target(dimension):
    """Separating API generation must not admit storage settings that consume the whole memory target."""
    policy = _settings().conversation_memory_policy()
    with pytest.raises(ValidationError, match=f"summary_max_{dimension} must be below low_water_{dimension}"):
        _settings(**{
            "summary_max_tokens": policy.low_water_tokens + 1,
            f"memory_summary_max_{dimension}": getattr(policy, f"low_water_{dimension}"),
        })
