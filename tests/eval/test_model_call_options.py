"""Evaluator requests must respect the configured model's sampling support."""
from __future__ import annotations

import json
import socket

import httpx
import pytest
from langchain_openai import ChatOpenAI

from src.core.answer_schema import finalize_answer, text_document
from src.eval.config_models import BenchmarkCase
from src.eval.judge_llm import LLMJudge
from src.eval.request_contract_semantics import build_semantic_assessor
from src.infra.settings import AppSettings


@pytest.fixture
def provider(monkeypatch):
    requests = []
    output = {}

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if payload["model"].startswith("gpt-6") and payload.get("temperature", 1) != 1:
            return httpx.Response(400, json={"error": {
                "message": "Only the default temperature (1) is supported.",
                "type": "invalid_request_error",
                "param": "temperature",
                "code": "unsupported_value",
            }})
        return httpx.Response(200, json={
            "id": "chatcmpl-options", "object": "chat.completion", "created": 1,
            "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(output),
            }}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        })

    def deny_network(*args, **kwargs):
        raise AssertionError("Model option tests must use the HTTP fake.")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        def create_model(**kwargs):
            return ChatOpenAI(**(kwargs | {
                "api_key": "test-key",
                "base_url": "https://model-options.invalid/v1",
                "http_client": client,
            }))

        monkeypatch.setattr("src.eval.judge_llm.ChatOpenAI", create_model)
        monkeypatch.setattr("src.eval.request_contract_semantics.ChatOpenAI", create_model)
        yield requests, output


@pytest.mark.parametrize(("model", "temperature"), [
    ("gpt-6-luna", None),
    ("gpt-6.1-sol", None),
    ("gpt-5.6-luna", None),
    ("gpt-4.1", 0),
])
def test_judge_scores_with_supported_wire_options(provider, model, temperature):
    requests, output = provider
    output.update({
        "score": 0.9, "reason": "The response answers the question.",
        "subscores": {name: 0.9 for name in (
            "answer_quality", "groundedness", "citation_traceability",
            "tool_choice", "format_language",
        )},
    })
    judge = LLMJudge(model_name=model)
    outcome = judge.score_case(
        case=BenchmarkCase(case_id="options", category="tool_action", query="Say hello."),
        tool_calls=[],
        response=finalize_answer(text_document("Hello."), []),
    )

    assert outcome.status == "succeeded", outcome.error
    assert outcome.score == 0.9
    assert outcome.subscores.answer_quality == 0.9
    assert len(requests) == 1
    assert requests[0]["model"] == model
    if temperature is None:
        assert "temperature" not in requests[0]
    else:
        assert requests[0]["temperature"] == temperature


@pytest.mark.parametrize(("model", "temperature"), [
    ("gpt-6-luna", None),
    ("gpt-4.1", 0),
])
def test_semantic_assessor_uses_supported_options_and_records_them(provider, monkeypatch, tmp_path, model, temperature):
    requests, output = provider
    output.update({
        name: {"status": "pass", "reason": "The requested meaning is preserved."}
        for name in ("subject", "operation", "target_language", "constraints")
    })
    monkeypatch.setenv("JUDGE_MODEL", model)
    monkeypatch.setenv("BENCHMARK_JUDGE_ENABLED", "true")
    monkeypatch.setattr("src.infra.settings.get_env_file_path", lambda: tmp_path / ".env")
    assessor = build_semantic_assessor(AppSettings(_env_file=None, openai_api_key="test-key"))
    result = assessor(
        {"query": "Explain caching.", "expected": {"body": {"kind": "compose"}}},
        {"contract_accepted": True, "canonical": {
            "body": {"kind": "compose", "instruction": "Explain caching."},
        }},
    )

    assert result["status"] == "evaluated", result.get("error")
    assert result["checks"]["subject"]["status"] == "pass"
    assert len(requests) == 1
    assert requests[0]["model"] == model
    assert assessor.definition["temperature"] == temperature
    if temperature is None:
        assert "temperature" not in requests[0]
    else:
        assert requests[0]["temperature"] == temperature
