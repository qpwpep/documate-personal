from __future__ import annotations

import json
import socket

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import PlannerState, RetrievalState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.infra.llm import build_llm_registry
from src.infra.settings import APP_ENV_SPEC_BY_NAME, AppSettings
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.graph_builder import build_agent_graph


def _evidence_packet(payload: dict) -> list[dict]:
    for message in payload.get("messages", payload.get("input", [])):
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content)
        if content.startswith("[Evidence Packet]"):
            return json.loads(content[content.index("[", len("[Evidence Packet]")):])
    return []


def _provider_response(payload: dict, content: str, *, finish_reason: str = "stop") -> dict:
    usage = {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130}
    if "input" in payload:
        status = "incomplete" if finish_reason == "length" else "completed"
        return {
            "id": "resp_test", "object": "response", "created_at": 1,
            "status": status, "model": payload["model"],
            "error": None, "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "output": [{
                "id": "msg_test", "type": "message", "role": "assistant", "status": status,
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }],
            "usage": usage,
        }
    return {
        "id": "chatcmpl_test", "object": "chat.completion", "created": 1,
        "model": payload["model"],
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
    }


@pytest.fixture
def provider(monkeypatch):
    """Only the external HTTP service is replaced; request construction and parsing are real."""
    for name in APP_ENV_SPEC_BY_NAME:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    requests: list[dict] = []
    behavior = {"timeout_once": False, "invalid_content": None, "finish_reason": "stop", "empty_output": False}

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if behavior["timeout_once"]:
            behavior["timeout_once"] = False
            raise httpx.ReadTimeout("provider timed out", request=request)
        packet = _evidence_packet(payload)
        if packet:
            document = text_document(packet[0]["excerpt"], basis="excerpt", refs=[packet[0]["id"]])
            content = document.model_dump_json()
        elif payload.get("response_format", {}).get("json_schema", {}).get("name") == "PlannerOutput":
            content = json.dumps({"use_retrieval": False, "tasks": [], "clarification_question": None})
        else:
            content = text_document("Hello.").model_dump_json()
        if behavior["invalid_content"] is not None:
            content = behavior["invalid_content"]
        response = _provider_response(payload, content, finish_reason=behavior["finish_reason"])
        if behavior["empty_output"]:
            response["output"] = []
        return httpx.Response(200, json=response)

    def deny_network(*args, **kwargs):
        raise AssertionError("Live network access is not allowed in token budget tests")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        def chat_model(**kwargs):
            return ChatOpenAI(**kwargs, http_client=client, base_url="https://token-budget.test/v1")

        monkeypatch.setattr("src.infra.llm.ChatOpenAI", chat_model)
        yield requests, behavior


def _settings(**overrides) -> AppSettings:
    values = {
        "openai_api_key": "test-key", "tavily_api_key": "test-key",
        "chat_model": "gpt-5.6-luna", "planner_model": "gpt-5.6-luna",
        "summary_model": "gpt-5.6-luna", "synthesis_reasoning_effort": "low",
        "synthesis_max_retries": 0, "verbose": False,
    }
    return AppSettings(_env_file=None, **(values | overrides))


def _state():
    text = "The configuration keeps the original source range. " * 100
    snapshot = build_snapshot(
        source_uri="https://docs.example.com/config", title="Configuration", media_type="text/plain",
        source_type="official", content=text, parser="test", parser_version="1",
    )
    evidence = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="config", kind="paragraph", text=text))
    task = RetrievalTask(route="docs", query="Explain the configuration", k=1)
    hit = SearchHit(evidence=evidence, rank=1, requirement_id=task.requirement_id,
                    score=RetrievalScore(metric="test", raw=1, normalized=1, direction="higher"))
    return build_graph_state_input(
        user_input=task.query, messages=[HumanMessage(content=task.query)],
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=[hit.model_dump(mode="json")]),
    )


def _synthesize(settings: AppSettings):
    registry = build_llm_registry(settings)
    return make_synthesize_node(
        registry.llm_synthesizer, registry.llm_synthesizer_compact,
        prompt_snippet_char_limit=settings.synthesis_prompt_snippet_chars,
        compact_prompt_snippet_char_limit=settings.synthesis_compact_prompt_snippet_chars,
    )(_state())


@pytest.mark.parametrize("responses_api", [False, True], ids=["chat", "responses"])
@pytest.mark.parametrize("output_cap", [1920, 4096])
def test_synthesis_output_cap_reaches_http_and_structured_citations_survive(provider, responses_api, output_cap):
    """Configured generation limits survive structured output while exact source citations remain valid."""
    requests, _ = provider
    settings = _settings(synthesis_use_responses_api=responses_api, synthesis_max_tokens=output_cap,
                         synthesis_prompt_snippet_chars=960)
    response = _synthesize(settings)["response"]
    assert len(requests) == 1
    payload = requests[0]
    cap_name = "max_output_tokens" if responses_api else "max_completion_tokens"
    assert {key: value for key, value in payload.items() if key in {"max_tokens", "max_completion_tokens", "max_output_tokens"}} == {cap_name: output_cap}
    schema = payload["text"]["format"] if responses_api else payload["response_format"]["json_schema"]
    assert schema["name"] == "AnswerDocument"
    assert schema["strict"] is True
    selected = response.evidence_packet[0]
    packet = _evidence_packet(payload)
    assert packet[0]["excerpt"] == selected.excerpt
    assert packet[0]["selection"] == selected.selection.model_dump(mode="json")
    assert response.result == finalize_answer(
        text_document(selected.excerpt, basis="excerpt", refs=[selected.id]), [selected], retrieval_required=True,
    )
    assert response.result.checks[0].support_status == "exact_match"


@pytest.mark.parametrize("responses_api", [False, True], ids=["chat", "responses"])
def test_timeout_uses_independent_compact_output_cap(provider, responses_api):
    """Changing the main output cap does not silently resize the compact recovery request."""
    requests, behavior = provider
    behavior["timeout_once"] = True
    settings = _settings(synthesis_use_responses_api=responses_api, synthesis_max_tokens=4096,
                         synthesis_compact_max_tokens=317, synthesis_prompt_snippet_chars=1800)
    response = _synthesize(settings)["response"]
    cap_name = "max_output_tokens" if responses_api else "max_completion_tokens"
    assert [request[cap_name] for request in requests] == [4096, 317]
    assert len(_evidence_packet(requests[0])[0]["excerpt"]) > 900
    selected = response.evidence_packet[0]
    assert len(selected.excerpt) <= 900
    assert response.result == finalize_answer(
        text_document(selected.excerpt, basis="excerpt", refs=[selected.id]), [selected], retrieval_required=True,
    )


def test_output_and_prompt_candidates_change_independent_http_fields(provider):
    """Output-only tuning preserves the prompt, and prompt-only tuning preserves the generation cap."""
    requests, _ = provider
    for output_cap, snippet_chars in [(1920, 960), (4096, 960), (1920, 1800), (4096, 1800)]:
        _synthesize(_settings(synthesis_use_responses_api=False, synthesis_max_tokens=output_cap,
                              synthesis_prompt_snippet_chars=snippet_chars))
    baseline, output_only, prompt_only, combined = requests
    assert baseline["messages"] == output_only["messages"]
    assert prompt_only["messages"] == combined["messages"]
    assert [request["max_completion_tokens"] for request in requests] == [1920, 4096, 1920, 4096]
    assert len(_evidence_packet(prompt_only)[0]["excerpt"]) > len(_evidence_packet(baseline)[0]["excerpt"])
    assert {k: v for k, v in baseline.items() if k != "messages"} == {k: v for k, v in prompt_only.items() if k != "messages"}


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_invalid_or_truncated_provider_json_stays_a_grounded_fallback(provider, finish_reason):
    """Broken provider output cannot become user-visible text or trigger an unbounded generation retry."""
    requests, behavior = provider
    behavior.update(invalid_content='{"blocks": ["UNVALIDATED', finish_reason=finish_reason)
    updates = _synthesize(_settings(synthesis_use_responses_api=False, synthesis_max_tokens=4096))
    result = updates["response"].result
    assert len(requests) == 1
    assert "UNVALIDATED" not in export_answer_text(result)
    assert result.citations
    assert any(check.support_status == "exact_match" for check in result.checks)
    assert updates["debug"].synthesis_errors


def test_planner_output_cap_reaches_http_without_synthesis_settings(provider):
    """Planning retains its own generation limit and strict structured response independently of synthesis."""
    requests, _ = provider
    registry = build_llm_registry(_settings(planner_max_tokens=654, synthesis_max_tokens=4096))
    result = registry.llm_planner.invoke([HumanMessage(content="Hello")])
    assert requests[0]["max_completion_tokens"] == 654
    assert requests[0]["response_format"]["json_schema"]["name"] == "PlannerOutput"
    assert PlannerOutput.model_validate(result["parsed"]) == PlannerOutput(use_retrieval=False, tasks=[])


@pytest.mark.parametrize("empty_output", [False, True], ids=["truncated-json", "no-visible-output"])
def test_responses_output_limit_exhaustion_keeps_a_grounded_fallback(provider, empty_output):
    """Exhausting a Responses generation budget cannot expose partial JSON or trigger an extra attempt."""
    requests, behavior = provider
    behavior.update(invalid_content='{"blocks": ["UNVALIDATED', finish_reason="length", empty_output=empty_output)
    updates = _synthesize(_settings(synthesis_use_responses_api=True, synthesis_max_tokens=4096))
    result = updates["response"].result
    assert len(requests) == 1
    assert requests[0]["max_output_tokens"] == 4096
    assert "UNVALIDATED" not in export_answer_text(result)
    assert result.citations
    assert any(check.support_status == "exact_match" for check in result.checks)
    assert updates["debug"].synthesis_errors


@pytest.mark.parametrize("responses_api", [False, True], ids=["chat", "responses"])
def test_compiled_graph_preserves_role_specific_output_settings(provider, responses_api):
    """The application graph carries each role's configured limit through to a valid answer."""
    requests, _ = provider
    graph = build_agent_graph(_settings(synthesis_use_responses_api=responses_api,
                                       planner_max_tokens=654, synthesis_max_tokens=2081))
    result = graph.invoke(build_graph_state_input(user_input="Hello"))
    assert len(requests) == 2
    assert requests[0]["max_completion_tokens"] == 654
    assert requests[1]["max_output_tokens" if responses_api else "max_completion_tokens"] == 2081
    assert export_answer_text(result["response"].result) == "Hello."
