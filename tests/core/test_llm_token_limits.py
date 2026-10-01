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
from src.core.request_contracts import RequestContract, WireRequestContract
from src.infra.llm import build_llm_registry
from src.infra.settings import APP_ENV_SPEC_BY_NAME, AppSettings
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.graph_builder import build_agent_graph
from tests.synthesis_fixtures import synthesis_excerpt_limits


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
    behavior = {"timeout_once": False, "invalid_content": None, "finish_reason": "stop", "empty_output": False,
                "request_paths": []}

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        behavior["request_paths"].append(request.url.path)
        if behavior["timeout_once"]:
            behavior["timeout_once"] = False
            raise httpx.ReadTimeout("provider timed out", request=request)
        packet = _evidence_packet(payload)
        if packet:
            document = text_document(packet[0]["excerpt"], basis="excerpt", refs=[packet[0]["id"]])
            content = document.model_dump_json()
        elif payload.get("response_format", {}).get("json_schema", {}).get("name") == "PlannerOutput":
            content = json.dumps({
                "use_retrieval": False, "tasks": [],
                "request_contract": WireRequestContract(slack_recipient={"state": "omitted"}).model_dump(mode="json"),
            })
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
        request_contract=RequestContract(request_id="token-budget-request"),
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=[hit.model_dump(mode="json")]),
    )


def _synthesize(settings: AppSettings):
    registry = build_llm_registry(settings)
    return make_synthesize_node(
        registry.llm_synthesizer, registry.llm_synthesizer_compact,
        excerpt_limits=synthesis_excerpt_limits(normal_chars=settings.synthesis_prompt_snippet_chars, compact_chars=settings.synthesis_compact_prompt_snippet_chars),
    )(_state())




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




















