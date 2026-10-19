"""Exercise the Decisions boundary through a real SDK and fake HTTP provider."""
import json
import logging

import httpx2 as httpx
import pytest
from openai import OpenAI

from src.core.decisions import DecisionResult
from src.core.llm_errors import LLMCallError
from src.infra.decisions import DecisionsClient
from src.infra.llm_boundary import capture_llm_diagnostics, request_call_budget
from src.runtime.agent_runtime.llm_usage import capture_llm_usage


STAGE = "pre_synthesis_validation"
QUESTION = {"type": "predicate", "name": "ready", "instructions": "Is the evidence sufficient?"}


def decision(*, answers=None, **overrides):
    return httpx.Response(200, headers={"x-request-id": "req_decisions_test"}, json={
        "model": "gpt-6-luna",
        "answers": [{"type": "predicate", "name": "ready", "probability": 0.9}] if answers is None else answers,
        "usage": {"input_tokens": 42, "output_tokens": 0, "total_tokens": 42},
        **overrides,
    })


@pytest.fixture
def invoke(monkeypatch):
    monkeypatch.setattr("src.infra.decisions.time.sleep", lambda _delay: None)

    def run(replies, *, questions=None, repeat=1, stage=STAGE):
        seen = []

        def handle(request):
            seen.append(request)
            reply = replies[min(len(seen) - 1, len(replies) - 1)]
            if isinstance(reply, Exception):
                raise reply
            return reply(request) if callable(reply) else reply

        with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
            sdk = OpenAI(api_key="test-secret", base_url="https://provider.test/v1", http_client=http_client,
                         max_retries=0)
            client = DecisionsClient(sdk, model="gpt-6-luna", timeout_seconds=20)
            with capture_llm_usage() as usage, capture_llm_diagnostics() as diagnostics:
                try:
                    for _ in range(repeat):
                        result = client.evaluate(input="private evidence", questions=questions or [QUESTION], stage=stage)
                except LLMCallError as exc:
                    result = exc
                return result, seen, usage.snapshot(), diagnostics

    return run


def test_evaluate_posts_predicates_to_decisions_and_records_real_usage(invoke, caplog):
    caplog.set_level(logging.INFO)
    result, seen, usage, diagnostics = invoke([decision()])
    assert result == DecisionResult(probabilities={"ready": 0.9}, refused_names=set())
    assert len(seen) == 1
    assert seen[0].url.path == "/v1/decisions"
    assert json.loads(seen[0].content) == {
        "model": "gpt-6-luna", "input": "private evidence", "questions": [QUESTION],
    }
    assert seen[0].extensions["timeout"]["read"] <= 20
    assert len(usage) == 1
    assert usage[0].stage == STAGE
    assert usage[0].path == "decisions"
    assert usage[0].model_name == "gpt-6-luna"
    assert usage[0].usage.input_tokens == 42
    assert usage[0].usage.output_tokens == 0
    assert usage[0].response_metadata["endpoint"] == "decisions"
    assert diagnostics == []
    assert "req_decisions_test" in caplog.text
    assert "private evidence" not in caplog.text
    assert "test-secret" not in caplog.text


def test_later_batch_failure_does_not_return_partial_validation(invoke):
    questions = [{**QUESTION, "name": f"requirement_{index}"} for index in range(201)]

    def first_batch(request):
        return decision(answers=[{"type": "predicate", "name": question["name"], "probability": 1}
                                 for question in json.loads(request.content)["questions"]])

    result, seen, usage, _ = invoke([first_batch, decision(answers=[])], questions=questions)
    assert isinstance(result, LLMCallError)
    assert result.problem.code == "model_output_invalid"
    assert len(seen) == 2
    assert len(usage) == 2


def test_refusal_preserves_successes_across_batches(invoke):
    refused_names = {"requirement_0", "requirement_200"}
    questions = [{**QUESTION, "name": f"requirement_{index}"} for index in range(201)]

    def reply(request):
        return decision(answers=[
            {"type": "refusal", "name": question["name"]}
            if question["name"] in refused_names else
            {"type": "predicate", "name": question["name"], "probability": 0.8}
            for question in reversed(json.loads(request.content)["questions"])
        ])

    result, seen, usage, diagnostics = invoke([reply], questions=questions)
    assert result == DecisionResult(
        probabilities={question["name"]: 0.8 for question in questions
                       if question["name"] not in refused_names},
        refused_names=refused_names,
    )
    assert [len(json.loads(request.content)["questions"]) for request in seen] == [200, 1]
    assert len(usage) == 2
    assert diagnostics == []


@pytest.mark.parametrize("answers", [
    [{"type": "predicate", "name": "wrong", "probability": 0.9}],
    [{"type": "choice", "name": "ready", "probability": 0.9}],
    [{"type": "predicate", "name": "ready", "probability": "0.9"}],
    [{"type": "predicate", "name": "ready", "probability": True}],
    [{"type": "predicate", "name": "ready", "probability": -0.1}],
    [{"type": "predicate", "name": "ready", "probability": 1.1}],
])
def test_invalid_answers_fail_closed_without_repair_or_payload_logging(invoke, answers, caplog):
    result, seen, usage, diagnostics = invoke([decision(answers=answers)])
    assert isinstance(result, LLMCallError)
    assert result.problem.code == "model_output_invalid"
    assert len(seen) == 1
    assert usage[0].usage.input_tokens == 42
    assert diagnostics[0].endpoint == "decisions"
    assert diagnostics[0].recovery == "stop"
    assert "private evidence" not in caplog.text


@pytest.mark.parametrize("content", [
    '{"answers":[{"type":"predicate","name":"ready","probability":NaN}]}',
    '{"answers":[{"type":"predicate","name":"ready","probability":0.1,"probability":0.9}]}',
    'invalid JSON',
    '[]',
])
def test_malformed_json_or_duplicate_fields_fail_closed(invoke, content):
    result, seen, _, _ = invoke([httpx.Response(200, text=content)])
    assert isinstance(result, LLMCallError)
    assert result.problem.code == "model_output_invalid"
    assert len(seen) == 1


def test_mixed_refusal_retains_individual_probabilities_without_refusal_text(invoke, caplog):
    caplog.set_level(logging.INFO)
    questions = [{**QUESTION, "name": name} for name in ("first", "blocked", "last")]
    result, seen, usage, diagnostics = invoke([decision(answers=[
        {"type": "predicate", "name": "first", "probability": 0.95},
        {"type": "refusal", "name": "blocked", "refusal": "private provider refusal"},
        {"type": "predicate", "name": "last", "probability": 0.2},
    ])], questions=questions)
    assert not isinstance(result, LLMCallError)
    assert result.probabilities == {"first": 0.95, "last": 0.2}
    assert result.refused_names == {"blocked"}
    assert len(seen) == 1
    assert usage[0].usage.total_tokens == 42
    assert diagnostics == []
    assert "probability_count=2" in caplog.text
    assert "refusal_count=1" in caplog.text
    assert "private provider refusal" not in repr(result)
    assert "private provider refusal" not in caplog.text


@pytest.mark.parametrize("answers", [
    [{"type": "refusal", "name": "first"},
     {"type": "predicate", "name": "last", "probability": "invalid"}],
    [{"type": "refusal", "name": "first"},
     {"type": "predicate", "name": "first", "probability": 0.9}],
])
def test_malformed_answer_after_refusal_still_fails_the_whole_response(invoke, answers):
    questions = [{**QUESTION, "name": name} for name in ("first", "last")]
    result, seen, _, _ = invoke([decision(answers=answers)], questions=questions)
    assert isinstance(result, LLMCallError)
    assert result.problem.code == "model_output_invalid"
    assert len(seen) == 1


def test_transient_error_retries_same_request_with_observed_attempts(invoke):
    result, seen, usage, diagnostics = invoke([
        httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}}, headers={"retry-after": "0"}),
        decision(),
    ])
    assert result == DecisionResult(probabilities={"ready": 0.9}, refused_names=set())
    assert len(seen) == 2
    assert seen[0].content == seen[1].content
    assert [call.attempt for call in usage] == [1, 2]
    assert usage[0].usage.observation == "none"
    assert usage[1].usage.total_tokens == 42
    assert diagnostics[0].recovery == "retry"


def test_calls_share_the_stage_budget_across_retrieval_retries(invoke):
    result, seen, usage, diagnostics = invoke([decision()], repeat=4)
    assert result.problem.code == "call_budget_exhausted"
    assert len(seen) == 3
    assert len(usage) == 3
    assert diagnostics[-1].budget_reason == "calls"


def test_transient_failures_stop_after_three_requests_without_sdk_nested_retries(invoke):
    result, seen, usage, diagnostics = invoke([
        httpx.Response(503, json={"error": {"code": "server_error"}}, headers={"retry-after": "0"}),
    ])
    assert result.problem.code == "provider_unavailable"
    assert len(seen) == 3
    assert [record.attempt for record in usage] == [1, 2, 3]
    assert [diagnostic.recovery for diagnostic in diagnostics] == ["retry", "retry", "stop"]


def test_timeout_recovers_through_the_same_bounded_sdk_path(invoke):
    result, seen, usage, diagnostics = invoke([httpx.ReadTimeout("private evidence"), decision()])
    assert result == DecisionResult(probabilities={"ready": 0.9}, refused_names=set())
    assert len(seen) == 2
    assert usage[0].usage.observation == "none"
    assert diagnostics[0].transport_failure == "timeout"


def test_retry_after_exceeding_remaining_budget_does_not_sleep_or_retry(invoke):
    result, seen, _, diagnostics = invoke([
        httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}}, headers={"retry-after": "120"}),
    ])
    assert result.problem.code == "provider_rate_limited"
    assert len(seen) == 1
    assert diagnostics[0].recovery == "stop"


def test_permanent_authentication_failure_stops_and_redacts_provider_message(invoke, caplog):
    result, seen, _, diagnostics = invoke([httpx.Response(401, json={"error": {
        "message": "private evidence and test-secret", "code": "invalid_api_key", "type": "invalid_request_error",
    }})])
    assert result.problem.code == "provider_configuration"
    assert len(seen) == 1
    assert diagnostics[0].provider_status == 401
    assert diagnostics[0].endpoint == "decisions"
    assert "private evidence" not in caplog.text
    assert "test-secret" not in caplog.text
