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


@pytest.mark.parametrize("status,code,param,expected", [
    (400, "invalid_json_schema", "response_format", "provider_schema_invalid"),
    (400, "unsupported_parameter", "reasoning_effort", "provider_configuration"),
    (401, "invalid_api_key", None, "provider_configuration"),
    (429, "insufficient_quota", None, "provider_configuration"),
])
def test_permanent_failures_stop_without_repair_or_raw_error_exposure(invoke, status, code, param, expected):
    result, seen = invoke([httpx.Response(status, json={"error": {
        "message": "PRIVATE PROVIDER BODY", "type": "invalid_request_error", "code": code, "param": param,
    }}, headers={"x-request-id": "req-provider"})])
    assert isinstance(result, LLMCallError)
    assert result.problem.code == expected
    assert result.diagnostic.provider_request_id == "req-provider"
    assert result.diagnostic.schema_hash
    assert len(seen) == 1
    assert "PRIVATE" not in str(result)
    assert "PRIVATE" not in result.diagnostic.model_dump_json()


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


def test_repeated_invalid_output_is_not_a_clarification(invoke):
    result, seen = invoke([response('{"count":0}')])
    assert result.problem.code == "model_output_invalid"
    assert len(seen) == 2
    assert result.diagnostic.validation_paths


def test_graph_reentry_and_shared_budget_do_not_double_count_attempts(invoke):
    budget = CallBudget()
    with capture_llm_usage() as usage:
        first, _ = invoke([response()], budget=budget, attempt=1)
        second, _ = invoke([response()], budget=budget, attempt=2)
    assert first == second == Output(count=1)
    assert [call.attempt for call in usage.snapshot()] == [1, 2]


@pytest.mark.parametrize("reply,code", [
    (response(None, refusal="No"), "model_refusal"),
    (response('{"count":', finish="length"), "model_output_incomplete"),
])
def test_refusal_and_truncation_are_not_json_repairs(invoke, reply, code):
    result, seen = invoke([reply])
    assert result.problem.code == code
    assert len(seen) == 1


def test_total_attempt_budget_bounds_transport_and_output_repairs_together(invoke):
    busy = httpx.Response(503, json={"error": {"message": "busy", "type": "server_error"}})
    result, seen = invoke([busy, response('{"count":0}'), busy])
    assert result.problem.code == "provider_unavailable"
    assert len(seen) == 3


def test_wire_required_fields_are_checked_before_domain_defaults(invoke):
    class Defaults(BaseModel):
        model_config = ConfigDict(extra="forbid")
        label: str = "domain shorthand default"

    result, seen = invoke([response('{}'), response('{"label":"corrected"}')], model=Defaults)
    assert result == Defaults(label="corrected")
    assert len(seen) == 2


def test_wire_types_are_checked_before_pydantic_coercion(invoke):
    result, seen = invoke([response('{"count":"1"}'), response('{"count":2}')])
    assert result == Output(count=2)
    assert len(seen) == 2


def test_responses_content_filter_incompletion_is_a_refusal_without_repair(invoke):
    reply = httpx.Response(200, json={
        "id": "resp-test", "object": "response", "created_at": 1, "model": "gpt-4o-mini",
        "status": "incomplete", "incomplete_details": {"reason": "content_filter"},
        "output": [{"id": "msg-test", "type": "message", "role": "assistant", "status": "incomplete",
                    "content": [{"type": "output_text", "text": '{"count":1}', "annotations": []}]}],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    })
    result, seen = invoke([reply], responses_api=True)
    assert result.problem.code == "model_refusal"
    assert len(seen) == 1


@pytest.mark.parametrize("budget,reason", [
    (CallBudget(max_calls=0), "calls"),
    (CallBudget(timeout_seconds=0), "deadline"),
])
def test_exhausted_budget_preserves_its_actual_cause_without_claiming_invalid_model_output(invoke, budget, reason):
    with capture_llm_diagnostics() as diagnostics:
        result, seen = invoke([response()], budget=budget)
    assert result.problem.code == "call_budget_exhausted"
    assert result.diagnostic.budget_reason == reason
    assert seen == []
    assert diagnostics[-1] == result.diagnostic


def test_validator_programming_errors_do_not_retry_or_blame_model_output(invoke):
    def broken_validator(_value):
        raise TypeError("INTERNAL PRIVATE DETAIL")

    result, seen = invoke([response()], validator=broken_validator)
    assert result.problem.code == "internal_error"
    assert len(seen) == 1
    assert "PRIVATE" not in result.diagnostic.model_dump_json()


def test_generated_extra_keys_do_not_leak_into_safe_diagnostics(invoke):
    with capture_llm_diagnostics() as diagnostics:
        result, seen = invoke([response('{"count":1,"PRIVATE_GENERATED_SECRET":1}'), response()])
    assert result == Output(count=1)
    assert len(seen) == 2
    assert diagnostics
    assert "PRIVATE_GENERATED_SECRET" not in diagnostics[0].model_dump_json()


def test_quota_type_without_a_code_is_permanent_configuration_failure(invoke):
    result, seen = invoke([httpx.Response(429, json={"error": {
        "type": "insufficient_quota", "code": None, "message": "quota exhausted",
    }})])
    assert result.problem.code == "provider_configuration"
    assert len(seen) == 1


def test_diagnostics_include_compiler_and_sdk_versions(invoke):
    result, _seen = invoke([httpx.Response(400, json={"error": {"code": "invalid_json_schema"}})])
    assert result.diagnostic.schema_compiler_version
    assert result.diagnostic.openai_sdk_version
    assert result.diagnostic.langchain_openai_version


def test_timeout_retry_can_be_reserved_for_the_compact_attempt(invoke):
    from src.infra.llm_boundary import is_timeout_failure

    result, seen = invoke([httpx.ReadTimeout("timeout")], retry_timeouts=False)
    assert result.problem.code == "provider_unavailable"
    assert is_timeout_failure(result)
    assert len(seen) == 1


def test_http_unavailability_still_retries_when_timeout_recovery_is_external(invoke):
    result, seen = invoke([httpx.Response(503, headers={"Retry-After": "0"}, json={"error": {"code": "server_error"}}), response()], retry_timeouts=False)
    assert result == Output(count=1)
    assert len(seen) == 2


def test_retry_after_larger_than_remaining_budget_prevents_another_request(invoke):
    result, seen = invoke([httpx.Response(429, headers={"Retry-After": "60"}, json={"error": {
        "code": "rate_limit_exceeded", "message": "busy",
    }}), response()], budget=CallBudget(timeout_seconds=5))
    assert result.problem.code == "provider_rate_limited"
    assert result.problem.retry_after_seconds == 60
    assert result.diagnostic.recovery == "stop"
    assert len(seen) == 1


def test_remaining_budget_reaches_the_real_http_client_timeout(invoke):
    observed = []
    result, _seen = invoke([response()], budget=CallBudget(timeout_seconds=5),
        request_observer=lambda request: observed.append(request.extensions["timeout"]))
    assert result == Output(count=1)
    assert observed
    assert all(0 < seconds <= 5 for seconds in observed[0].values())


def test_typed_responses_sdk_error_preserves_provider_classification(invoke):
    with capture_llm_diagnostics() as diagnostics:
        result, seen = invoke([response_api(error={"code": "server_error", "message": "PRIVATE PROVIDER ERROR"}), response_api()], responses_api=True)
    assert result == Output(count=1)
    assert len(seen) == 2
    assert diagnostics[0].code == "provider_unavailable"
    assert diagnostics[0].provider_code == "server_error"
    assert "PRIVATE" not in diagnostics[0].model_dump_json()


def test_duplicate_json_keys_are_invalid_even_when_the_last_value_would_validate(invoke):
    result, seen = invoke([response('{"count":0,"count":1}'), response('{"count":2}')])
    assert result == Output(count=2)
    assert len(seen) == 2


def test_nonstandard_json_numbers_cannot_pass_through_wire_validation(invoke):
    class Number(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: float

    result, seen = invoke([response('{"value":NaN}'), response('{"value":1.5}')], model=Number)
    assert result == Number(value=1.5)
    assert len(seen) == 2
