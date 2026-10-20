"""Model options must remain valid in the actual serialized provider request."""
from __future__ import annotations

import json
import socket

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.core.planner_schema import InitialPlannerOutput
from src.infra.llm import build_llm_registry
from src.infra.settings import APP_ENV_SPEC_BY_NAME, AppSettings


_CLIENTS = [
    pytest.param("llm_synthesizer", False, id="synthesis-chat"),
    pytest.param("llm_synthesizer_compact", False, id="compact-chat"),
    pytest.param("llm_synthesizer", True, id="synthesis-responses"),
    pytest.param("llm_synthesizer_compact", True, id="compact-responses"),
    pytest.param("llm_planner", False, id="planner"),
    pytest.param("llm_summarizer", False, id="summary"),
]


def _provider_response(payload: dict, *, responses_api: bool) -> dict:
    content = "The request succeeded."
    if "response_format" in payload:
        content = InitialPlannerOutput.model_validate({
            "use_retrieval": False,
            "tasks": [],
            "request_contract": {
                "body": {"kind": "compose", "instruction": "Explain the cache."},
                "slack_recipient": {"state": "omitted"},
            },
        }).model_dump_json()
    if responses_api:
        return {
            "id": "resp-model-options", "object": "response", "created_at": 1,
            "status": "completed", "model": payload["model"],
            "output": [{
                "id": "msg-model-options", "type": "message", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }],
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        }
    return {
        "id": "chatcmpl-model-options", "object": "chat.completion", "created": 1,
        "model": payload["model"],
        "choices": [{
            "index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }


@pytest.fixture
def provider(monkeypatch):
    """Keep model construction and SDK serialization real; replace only HTTP."""
    for name in APP_ENV_SPEC_BY_NAME:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    def deny_network(*_args, **_kwargs):
        raise AssertionError("Model option tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    requests: list[tuple[str, dict]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if payload["model"].startswith("gpt-6") and payload.get("temperature", 1) != 1:
            return httpx.Response(400, json={"error": {
                "message": "Only the default temperature (1) is supported.",
                "type": "invalid_request_error",
                "param": "temperature",
                "code": "unsupported_value",
            }})
        return httpx.Response(200, json=_provider_response(
            payload, responses_api=request.url.path == "/v1/responses",
        ))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        def chat_model(**kwargs):
            return ChatOpenAI(
                **kwargs, http_client=client, base_url="https://model-options.test/v1",
            )

        monkeypatch.setattr("src.infra.llm.ChatOpenAI", chat_model)
        yield requests


def _invoke_client(model_name: str, client_name: str, responses_api: bool, *, reasoning=None):
    settings = AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test-key",
        chat_model=model_name, planner_model=model_name, summary_model=model_name,
        synthesis_use_responses_api=responses_api,
        synthesis_reasoning_effort=reasoning, planner_reasoning_effort=reasoning,
        verbose=False,
    )
    registry = build_llm_registry(settings)
    result = getattr(registry, client_name).invoke([HumanMessage(content="Explain the cache.")])
    if client_name == "llm_planner":
        assert result["parsing_error"] is None
        assert result["parsed"]["request_contract"]["body"]["instruction"] == "Explain the cache."
    else:
        assert result.content


@pytest.mark.parametrize(("client_name", "responses_api"), _CLIENTS)
@pytest.mark.parametrize(
    ("model_name", "expected_temperature"),
    [
        pytest.param("gpt-6-luna", None, id="gpt6-default"),
        pytest.param("gpt-5.6-luna", None, id="gpt5-existing-default"),
        pytest.param("gpt-4o-mini", 0, id="legacy-zero"),
    ],
)
def test_registry_clients_send_supported_temperature(
    provider, client_name, responses_api, model_name, expected_temperature,
):
    _invoke_client(model_name, client_name, responses_api)

    assert len(provider) == 1
    path, payload = provider[0]
    assert path == ("/v1/responses" if responses_api else "/v1/chat/completions")
    assert payload["model"] == model_name
    if expected_temperature is None:
        assert "temperature" not in payload
    else:
        assert payload["temperature"] == expected_temperature


@pytest.mark.parametrize(
    ("client_name", "responses_api"),
    [
        pytest.param("llm_synthesizer", False, id="synthesis-chat"),
        pytest.param("llm_synthesizer", True, id="synthesis-responses"),
        pytest.param("llm_planner", False, id="planner"),
    ],
)
def test_gpt5_explicit_no_reasoning_keeps_existing_zero_temperature(
    provider, client_name, responses_api,
):
    _invoke_client("gpt-5.6-luna", client_name, responses_api, reasoning="none")

    assert len(provider) == 1
    _, payload = provider[0]
    assert payload["temperature"] == 0
    if responses_api:
        assert payload["reasoning"]["effort"] == "none"
    else:
        assert payload["reasoning_effort"] == "none"
