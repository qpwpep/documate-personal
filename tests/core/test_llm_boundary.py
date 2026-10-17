import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.core.llm_errors import LLMCallError
from src.infra.llm_boundary import CallBudget, bind_structured_output, run_structured_call


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = Field(ge=1)


def response(content='{"count":1}'):
    return httpx.Response(200, json={
        "id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": content,
        }}], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    })


@pytest.fixture
def invoke(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr("src.infra.llm_boundary.time.sleep", lambda _seconds: None)

    def run(replies):
        seen = []
        def handle(request):
            seen.append(json.loads(request.content))
            value = replies[min(len(seen) - 1, len(replies) - 1)]
            if isinstance(value, Exception):
                raise value
            return value
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            llm = bind_structured_output(ChatOpenAI(
                model="gpt-4o-mini", api_key="test", http_client=client,
                base_url="https://provider.test/v1", max_retries=0,
                use_responses_api=False,
            ), Output)
            try:
                result = run_structured_call(llm, [HumanMessage(content="Count one")],
                    stage="planner", validate=Output.model_validate,
                    budget=CallBudget())
            except LLMCallError as exc:
                result = exc
        return result, seen
    return run


def test_temporary_failure_retries_the_same_prompt_then_returns_validated_output(invoke):
    result, seen = invoke([httpx.Response(429, json={"error": {
        "message": "busy", "type": "rate_limit_error", "code": "rate_limit_exceeded",
    }}, headers={"retry-after": "0"}), response()])
    assert result == Output(count=1)
    assert len(seen) == 2
    assert seen[0]["messages"] == seen[1]["messages"]


def test_invalid_output_gets_one_bounded_repair_and_keeps_original_input(invoke):
    result, seen = invoke([response('{"count":0}'), response()])
    assert result == Output(count=1)
    assert len(seen) == 2
    assert seen[1]["messages"][0] == seen[0]["messages"][0]
    assert "count" in seen[1]["messages"][-1]["content"]


def test_total_attempt_budget_bounds_transport_and_output_repairs_together(invoke):
    busy = httpx.Response(503, json={"error": {"message": "busy", "type": "server_error"}})
    result, seen = invoke([busy, response('{"count":0}'), busy])
    assert result.problem.code == "provider_unavailable"
    assert len(seen) == 3
