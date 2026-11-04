"""Whole-request deadlines preserve committed conversation memory."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.app.agent_manager import AgentFlowManager
from src.core.answer_schema import text_document
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.llm_boundary import remaining_request_seconds, request_deadline
from src.infra.settings import AppSettings
from tests.core.test_pending_request_delivery import delivery_tools


@pytest.fixture
def runtime_api(monkeypatch):
    controls = {"expire_at": None, "deadline": None, "summary_rate_limited": False}
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.path.endswith("/decisions"):
            stage = "validation"
            reply = {"model": "gpt-6-luna", "answers": [
                {"type": "predicate", "name": question["name"], "probability": 0.99}
                for question in payload["questions"]
            ], "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32}}
        else:
            schema = payload.get("response_format", {}).get("json_schema", {}).get("schema", {})
            if "use_retrieval" in schema.get("properties", {}):
                stage = "planner"
                context = next(message["content"] for message in payload["messages"]
                               if message.get("name") == "request_context")
                turn = json.loads(context.split("\n", 1)[1])["user_turn_ledger"][-1]
                contract = WireRequestContract.model_validate({
                    "actions": {"save_text": {"intent": "requested", "evidence_ids": ["request"]}}
                    if "저장" in turn["text"] else {},
                    "evidence": [{"id": "request", "turn_id": turn["turn_id"], "quote": turn["text"],
                                  "scope": "current_request", "interpretation": "instruction"}],
                    "slack_recipient": {"state": "omitted"},
                })
                content = InitialPlannerOutput(use_retrieval=False, tasks=[], request_contract=contract).model_dump_json()
            elif schema:
                stage = "synthesis"
                content = text_document("검증된 답변입니다.").model_dump_json()
            else:
                stage = "summary"
                content = "이전 대화의 요약입니다."
            reply = {
                "id": "chat-deadline", "object": "chat.completion", "created": 1, "model": payload["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
        requests.append({"stage": stage, "remaining": remaining_request_seconds(),
                         "timeout": request.extensions["timeout"]})
        if controls["expire_at"] == stage:
            controls["deadline"].expires_at = 0
        if controls["summary_rate_limited"] and stage == "summary":
            return httpx.Response(429, headers={"retry-after": "30"}, json={"error": {
                "message": "Please retry later", "type": "rate_limit_error", "code": "rate_limit_exceeded",
            }})
        return httpx.Response(200, json=reply)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **kwargs: ChatOpenAI(
            **kwargs, http_client=client, base_url="https://provider.test/v1",
        ))
        monkeypatch.setattr("src.infra.llm.OpenAI", lambda **kwargs: OpenAI(
            **kwargs, http_client=client, base_url="https://provider.test/v1",
        ))
        yield controls, requests


def _settings(**overrides):
    return AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test-key",
        chat_model="gpt-4o-mini", planner_model="gpt-4o-mini", summary_model="gpt-4o-mini",
        synthesis_use_responses_api=False, request_timeout_seconds=10, **overrides,
    )


@pytest.mark.parametrize("late_stage", ["planner", "synthesis", "validation"])
def test_late_provider_success_does_not_publish_answer_save_or_commit_memory(
    runtime_api, delivery_tools, late_stage,
):
    controls, requests = runtime_api
    _save, _slack, deliveries, output = delivery_tools
    manager = AgentFlowManager(_settings())
    stable = manager.run_agent_flow("짧은 답변을 작성해줘")
    assert stable.outcome.status == "completed"
    before = manager.session.snapshot_conversation_memory()
    previous = manager.session.previous_response.model_copy(deep=True)
    controls["expire_at"] = late_stage

    with request_deadline(10) as deadline:
        controls["deadline"] = deadline
        result = manager.run_agent_flow("답변을 작성하고 저장해줘")

    assert result.outcome.status == "failed"
    assert result.outcome.problem.code == "request_timeout"
    assert result.outcome.response is None
    assert result.outcome.answer_id is None
    assert manager.session.snapshot_conversation_memory() == before
    assert manager.session.previous_response == previous
    assert not output.exists()
    assert deliveries == []
    assert all(0 < item["remaining"] <= 10 for item in requests)
    assert all(0 < item["timeout"]["read"] <= 10 for item in requests)
    assert result.debug.llm_calls[-1].status == "failed"
    assert result.debug.llm_diagnostics[-1].code == "request_timeout"


def test_late_summary_does_not_fall_back_and_continue_to_planner(runtime_api, delivery_tools):
    controls, requests = runtime_api
    manager = AgentFlowManager(_settings(memory_high_water_turns=3, memory_low_water_turns=2))
    manager.session.commit_conversation_memory(
        messages=[HumanMessage(content="old request"), AIMessage(content="old answer"),
                  HumanMessage(content="recent request"), AIMessage(content="recent answer")],
        memory_summary="committed summary",
    )
    before = manager.session.snapshot_conversation_memory()
    controls["expire_at"] = "summary"
    with request_deadline(10) as deadline:
        controls["deadline"] = deadline
        result = manager.run_agent_flow("new request")

    assert result.outcome.problem.code == "request_timeout"
    assert result.outcome.response is None
    assert manager.session.snapshot_conversation_memory() == before
    assert [item["stage"] for item in requests] == ["summary"]


def test_summary_retry_after_outside_remaining_request_does_not_sleep(runtime_api, delivery_tools, monkeypatch):
    controls, requests = runtime_api
    controls["summary_rate_limited"] = True
    manager = AgentFlowManager(_settings(memory_high_water_turns=3, memory_low_water_turns=2))
    manager.session.commit_conversation_memory(
        messages=[HumanMessage(content="old request"), AIMessage(content="old answer"),
                  HumanMessage(content="recent request"), AIMessage(content="recent answer")],
        memory_summary="committed summary",
    )

    def reject_sleep(_seconds):
        pytest.fail("Retry-After exceeds the entire remaining request budget")

    monkeypatch.setattr("src.runtime.nodes.session.time", SimpleNamespace(sleep=reject_sleep))
    result = manager.run_agent_flow("new request")

    assert result.outcome.status == "completed"
    assert "committed summary" in manager.memory_summary
    assert [item["stage"] for item in requests].count("summary") == 1
    assert any(item.code == "provider_rate_limited" and item.recovery == "stop"
               for item in result.debug.llm_diagnostics)
