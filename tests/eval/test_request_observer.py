"""Observation must preserve the real planner's validation and time budget."""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_openai import ChatOpenAI

from src.core.llm_errors import LLMCallError
from src.core.planner_schema import InitialPlannerOutput
from src.eval.request_contract_eval import build_case_state, build_manifest, make_live_observer
from src.infra.llm_boundary import bind_structured_output, capture_structured_calls
from src.infra.settings import AppSettings
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


def observe(monkeypatch, model):
    monkeypatch.setattr("src.infra.llm.build_llm_registry", lambda _settings: SimpleNamespace(llm_planner=model))
    return make_live_observer(SimpleNamespace(openai_api_key="test"))(deepcopy(CASE))


def test_observer_rejects_wire_missing_defaults_exactly_like_product(monkeypatch, planner_http):
    payload = planner_payload()
    del payload["aspect_references"]  # Pydantic has a default; strict wire schema requires the field.
    product, product_requests = planner_http([payload])
    state, _ = build_case_state(CASE)
    with pytest.raises(LLMCallError) as product_error:
        make_planner_node(product, verbose=False)(state)
    assert product_error.value.problem.code == "model_output_invalid"
    evaluated, observed_requests = planner_http([payload])
    observation = observe(monkeypatch, evaluated)
    assert observation["contract_accepted"] is False
    assert len(observed_requests) == len(product_requests) == 2
    assert len(observation["model_calls"]) == 2


def test_observer_forwards_same_bounded_timeout_as_product(monkeypatch, planner_http):
    payload = planner_payload()
    del payload["aspect_references"]
    product, product_requests = planner_http([payload])
    state, _ = build_case_state(CASE)
    with pytest.raises(LLMCallError):
        make_planner_node(product, verbose=False)(state)
    evaluated, observed_requests = planner_http([payload])
    observe(monkeypatch, evaluated)
    assert 0 < product_requests[0]["timeout"]["read"] <= 45
    assert 0 < observed_requests[0]["timeout"]["read"] <= 45


def test_observer_preserves_contract_candidates_from_all_repair_attempts(monkeypatch, planner_http):
    missing_contract = {"use_retrieval": False, "tasks": [], "aspect_references": []}
    unauthorized = planner_payload()
    unauthorized["request_contract"]["actions"]["slack_notify"]["intent"] = "requested"
    model, requests = planner_http([missing_contract, unauthorized])
    observation = observe(monkeypatch, model)
    assert observation["contract_accepted"] is False
    assert len(requests) == len(observation["model_calls"]) == 2
    assert [item["status"] for item in observation["raw_candidates"]] == ["missing_contract", "available"]
    assert observation["raw_candidates"][1]["candidate"]["actions"]["slack_notify"]["intent"] == "requested"
    assert [item["attempt"] for item in observation["raw_candidates"]] == [1, 2]


def test_observer_marks_invalid_json_as_unobserved_instead_of_accepting_tolerant_parse(monkeypatch, planner_http):
    payload = json.dumps(planner_payload())[:-1]
    model, _ = planner_http([payload])
    observation = observe(monkeypatch, model)
    assert observation["contract_accepted"] is False
    assert [item["status"] for item in observation["raw_candidates"]] == ["invalid_json", "invalid_json"]
    assert all(item["candidate"] is None for item in observation["raw_candidates"])


def test_manifest_records_effective_reasoning_retry_and_budget_settings(monkeypatch):
    monkeypatch.setattr("src.eval.request_contract_eval.code_fingerprint", lambda: {"digest": "test", "files": {}})
    low = AppSettings(_env_file=None, openai_api_key="test", planner_reasoning_effort="low")
    high = low.model_copy(update={"planner_reasoning_effort": "high"})
    kwargs = {"run_id": "unit-test", "cases": [], "policy": {"policy_id": "unit"},
              "repeats": 1, "max_workers": 1, "mode": "test"}
    first = build_manifest(settings=low, **kwargs)
    second = build_manifest(settings=high, **kwargs)
    assert first["hashes"]["settings"] != second["hashes"]["settings"]
    assert first["settings"]["planner_reasoning_effort"] == "low"
    assert first["settings"]["planner_provider_max_retries"] == 0
    assert first["settings"]["planner_call_budget"] == {"max_calls": 3, "timeout_seconds": 45}
    assert first["hashes"]["semantic_assessor"] == first["semantic_assessor"]["digest"]
    assert first["scoring_contract_version"] == "request-contract-semantics-v3"


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


def test_provider_failure_remains_an_unavailable_raw_attempt(monkeypatch, planner_http):
    model, requests = planner_http([httpx.Response(401, json={
        "error": {"message": "Invalid credential", "type": "authentication_error"},
    })])
    observation = observe(monkeypatch, model)
    assert observation["contract_accepted"] is False
    assert len(requests) == len(observation["model_calls"]) == 1
    assert observation["raw_candidates"] == [{"attempt": 1, "status": "unavailable", "candidate": None}]
    assert "exception" in observation["model_calls"][0]


def test_semantic_assessor_failure_keeps_accepted_product_and_raw_attempts(monkeypatch, planner_http):
    model, requests = planner_http([planner_payload()])
    monkeypatch.setattr("src.infra.llm.build_llm_registry", lambda _settings: SimpleNamespace(llm_planner=model))

    def unavailable(_case, _observation):
        raise ConnectionError("semantic evaluator unavailable")

    observation = make_live_observer(SimpleNamespace(openai_api_key="test"), body_semantic_assessor=unavailable)(deepcopy(CASE))
    assert observation["contract_accepted"] is True
    assert len(requests) == len(observation["raw_candidates"]) == 1
    assert observation["raw_candidates"][0]["candidate"]["body"]["kind"] == "compose"
    assert observation["body_semantics"]["status"] == "unavailable"
    assert observation["errors"] == []
