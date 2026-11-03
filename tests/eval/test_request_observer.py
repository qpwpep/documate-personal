"""Structured model observations stay scoped across repair attempts."""
from __future__ import annotations

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

from src.core.planner_schema import InitialPlannerOutput
from src.eval.request_contract_eval import build_case_state
from src.infra.llm_boundary import bind_structured_output, capture_structured_calls
from src.runtime.nodes.planner import make_planner_node


CASE = {"query": "Explain the Slack API.", "context": {}}


def planner_payload() -> dict:
    return InitialPlannerOutput.model_validate({
        "use_retrieval": False, "tasks": [],
        "request_contract": {
            "body": {"kind": "compose", "instruction": CASE["query"]},
            "slack_recipient": {"state": "omitted"},
        },
    }).model_dump(mode="json")


@pytest.fixture
def planner_http(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    clients = []

    def create(payloads):
        seen = []

        def handle(request):
            seen.append({"body": json.loads(request.content), "timeout": request.extensions["timeout"]})
            payload = payloads[min(len(seen) - 1, len(payloads) - 1)]
            if isinstance(payload, httpx.Response):
                return payload
            content = payload if isinstance(payload, str) else json.dumps(payload)
            return httpx.Response(200, json={
                "id": "chatcmpl-observer", "object": "chat.completion", "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": content,
                }}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            })

        client = httpx.Client(transport=httpx.MockTransport(handle), timeout=120)
        clients.append(client)
        model = bind_structured_output(ChatOpenAI(
            model="gpt-4o-mini", api_key="test", base_url="https://provider.test/v1",
            http_client=client, max_retries=0, use_responses_api=False,
        ), InitialPlannerOutput, name="PlannerOutput")
        return model, seen

    yield create
    for client in clients:
        client.close()


def test_capture_keeps_valid_repair_and_is_scoped_to_the_current_observer(planner_http):
    invalid = planner_payload()
    del invalid["aspect_references"]
    model, requests = planner_http([invalid, planner_payload()])
    state, _ = build_case_state(CASE)
    with capture_structured_calls() as outer:
        with capture_structured_calls() as inner:
            result = make_planner_node(model, verbose=False)(state)
        assert result["runtime"].request_contract.can_prepare_body()
        assert outer == []
        assert len(inner) == len(requests) == 2
        assert [record.attempt for record in inner] == [1, 2]
        assert "aspect_references" not in json.loads(inner[0].output["raw"].content)
        assert "aspect_references" in json.loads(inner[1].output["raw"].content)
        make_planner_node(model, verbose=False)(state)
    assert len(outer) == 1
    make_planner_node(model, verbose=False)(state)
    assert len(outer) == 1
