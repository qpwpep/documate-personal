import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.core.llm_errors import LLMCallError
from src.runtime.agent_runtime.llm_usage import capture_llm_usage
from src.infra.llm_boundary import (
    CallBudget, bind_structured_output, capture_llm_diagnostics, run_structured_call,
)


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = Field(ge=1)


def response(content='{"count":1}', *, finish="stop", refusal=None):
    return httpx.Response(200, json={
        "id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "finish_reason": finish, "message": {
            "role": "assistant", "content": content, "refusal": refusal,
        }}], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    })


def response_api(content='{"count":1}', *, error=None):
    return httpx.Response(200, json={
        "id": "resp-test", "object": "response", "created_at": 1, "model": "gpt-4o-mini",
        "status": "failed" if error else "completed", "error": error,
        "output": [] if error else [{"id": "msg-test", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": content, "annotations": []}]}],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    })


@pytest.fixture
def invoke(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr("src.infra.llm_boundary.time.sleep", lambda _seconds: None)

    def run(replies, *, model=Output, budget=None, validator=None, responses_api=False, request_observer=None, **options):
        seen = []
        def handle(request):
            seen.append(json.loads(request.content))
            if request_observer is not None:
                request_observer(request)
            value = replies[min(len(seen) - 1, len(replies) - 1)]
            if isinstance(value, Exception):
                raise value
            return value
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            llm = bind_structured_output(ChatOpenAI(
                model="gpt-4o-mini", api_key="test", http_client=client,
                base_url="https://provider.test/v1", max_retries=0,
                use_responses_api=responses_api,
            ), model)
            try:
                result = run_structured_call(llm, [HumanMessage(content="Count one")],
                    stage="planner", validate=validator or model.model_validate,
                    budget=budget or CallBudget(), **options)
            except LLMCallError as exc:
                result = exc
        return result, seen
    return run




@pytest.mark.parametrize("status", [429, 503])
def test_temporary_failure_retries_the_same_prompt_then_returns_validated_output(invoke, status):
    result, seen = invoke([httpx.Response(status, json={"error": {
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






























