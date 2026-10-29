import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.core.llm_errors import LLMCallError
from src.infra.llm_boundary import CallBudget, bind_structured_output, run_structured_call
from src.runtime.agent_runtime.llm_usage import capture_llm_usage


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = Field(default=1, ge=1)


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


def test_repair_preserves_failed_attempt_usage_and_records_both_outcomes(invoke):
    with capture_llm_usage() as recorder:
        result, _ = invoke([response('{"count":0}'), response()])
        calls = recorder.snapshot()

    assert result == Output(count=1)
    assert [call.status for call in calls] == ["failed", "completed"]
    assert [call.usage.total_tokens for call in calls] == [5, 5]
    assert all(call.model_name == "gpt-4o-mini" for call in calls)
    assert all(call.endpoint == "chat/completions" for call in calls)
    assert all(call.latency_ms is not None and call.latency_ms >= 0 for call in calls)
    assert calls[0].failure_type is not None
    assert calls[1].failure_type is None


def test_transport_retry_retains_failed_provider_identity_and_unknown_usage(invoke):
    with capture_llm_usage() as recorder:
        result, _ = invoke([httpx.Response(503, json={"error": {
            "message": "busy", "type": "server_error",
        }}, headers={"retry-after": "0"}), response()])
        calls = recorder.snapshot()

    assert result == Output(count=1)
    assert [call.status for call in calls] == ["failed", "completed"]
    assert calls[0].model_name == "gpt-4o-mini"
    assert calls[0].endpoint == "chat/completions"
    assert calls[0].usage.observation == "none"
    assert calls[0].latency_ms is not None and calls[0].latency_ms >= 0
    assert calls[0].failure_type is not None
    assert calls[1].usage.total_tokens == 5


@pytest.mark.parametrize("output", [None, 0])
def test_provider_missing_output_is_not_replaced_by_sdk_zero(invoke, output):
    payload = json.loads(response().content)
    payload["usage"] = {
        "prompt_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    }
    if output is not None:
        payload["usage"]["completion_tokens"] = output
    with capture_llm_usage() as recorder:
        result, _ = invoke([httpx.Response(200, json=payload)])
        observed = recorder.snapshot()[0].usage

    assert result == Output(count=1)
    assert observed.input_tokens == 100
    assert observed.output_tokens == output
    assert observed.total_tokens == (100 if output == 0 else None)
    assert observed.observation == ("complete" if output == 0 else "partial")


def test_truncated_completion_preserves_usage_carried_by_sdk_exception(invoke):
    payload = json.loads(response().content)
    payload["choices"][0]["finish_reason"] = "length"
    payload["usage"] = {
        "prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
        "prompt_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 30},
        "completion_tokens_details": {"reasoning_tokens": 10},
    }
    with capture_llm_usage() as recorder:
        result, _ = invoke([httpx.Response(200, json=payload)])
        call = recorder.snapshot()[0]

    assert isinstance(result, LLMCallError)
    assert call.status == "failed"
    assert call.usage.input_tokens == 100
    assert call.usage.output_tokens == 40
    assert call.usage.cached_input_tokens == 20
    assert call.usage.cache_write_input_tokens == 30
    assert call.usage.reasoning_output_tokens == 10


def test_total_attempt_budget_bounds_transport_and_output_repairs_together(invoke):
    busy = httpx.Response(503, json={"error": {"message": "busy", "type": "server_error"}})
    result, seen = invoke([busy, response('{"count":0}'), busy])
    assert result.problem.code == "provider_unavailable"
    assert len(seen) == 3


def test_unbound_collaborator_cannot_bypass_the_declared_wire_contract():
    class InvokeOnly:
        def invoke(self, messages):
            return {"count": "1"}

    with pytest.raises(TypeError, match="bound structured model"):
        run_structured_call(InvokeOnly(), [HumanMessage(content="Count one")],
                            stage="planner", validate=Output.model_validate)


@pytest.mark.parametrize("content", ["{}", '{"count":"1"}'])
def test_wire_defaults_and_types_are_checked_before_domain_coercion(invoke, content):
    result, seen = invoke([response(content)])
    assert result.problem.code == "model_output_invalid"
    assert len(seen) == 2


def test_remaining_call_budget_reaches_the_provider_transport():
    timeouts = []

    def handle(request):
        timeouts.append(request.extensions["timeout"]["read"])
        return response()

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        model = bind_structured_output(ChatOpenAI(
            model="gpt-4o-mini", api_key="test", http_client=client,
            base_url="https://provider.test/v1", max_retries=0, timeout=10,
            use_responses_api=False,
        ), Output)
        result = run_structured_call(model, [HumanMessage(content="Count one")],
            stage="planner", validate=Output.model_validate, budget=CallBudget(timeout_seconds=2))

    assert result.count == 1
    assert len(timeouts) == 1
    assert 0 < timeouts[0] <= 2
