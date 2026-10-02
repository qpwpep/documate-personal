"""Execution outcomes survive the HTTP and presentation boundaries without debug."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.app.web.agent_request_service import AgentRequestService
from src.app.web.app import create_app
from src.infra.sse import iter_sse_events
from tests.web.test_agent_request_service import _FakeCleaner, _request, _session_store


def _problem(code="provider_schema_invalid", next_action="fix_configuration"):
    return {
        "code": code, "stage": "planner", "message": "서비스 설정 오류로 요청을 처리하지 못했습니다.",
        "next_action": next_action,
    }


@pytest.mark.parametrize("status,problem,message,missing_slots", [
    ("failed", _problem(), "", []),
    ("refused", _problem("model_refusal", "none"), "", []),
    ("needs_input", None, "어느 문서를 사용할까요?", ["document"]),
])
def test_terminal_outcome_reaches_http_without_debug(status, problem, message, missing_slots):
    payload = {"status": status, "response": None, "problem": problem,
               "message": message, "missing_slots": missing_slots, "request_id": "untrusted-manager-id"}
    store = _session_store(payload)
    service = AgentRequestService(runtime_cleaner=_FakeCleaner(), session_store=store)
    app = create_app()
    app.state.agent_request_service = service
    request = _request(store, query="정상 질문입니다", session_id="result-session", include_debug=False)
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.post("/agent/stream", json=request.model_dump(mode="json"))
    finally:
        client.close()
    assert response.status_code == 200
    events = list(iter_sse_events([response.content]))
    terminals = [event.data for event in events if event.event == "final_response"]
    assert len(terminals) == 1
    final = terminals[0]
    assert final["status"] == status
    assert final["response"] is None
    assert final["message"] == message
    assert final["missing_slots"] == missing_slots
    assert final["debug"] is None
    if problem is None:
        assert final["problem"] is None
    else:
        assert final["problem"]["code"] == problem["code"]
    assert final["request_id"] == events[0].data["request_id"]
    assert final["request_id"] != "untrusted-manager-id"
    assert [event.event for event in events].count("error") == 0
    assert events[-1].event == "done"
    store.close_all()


@pytest.mark.parametrize("status,problem,message", [
    ("failed", _problem(), ""),
    ("refused", _problem("model_refusal", "none"), ""),
    ("needs_input", None, "어느 문서를 사용할까요?"),
])
@pytest.mark.parametrize("omit_provenance", [False, True])
def test_no_answer_outcome_does_not_require_answer_provenance(status, problem, message, omit_provenance):
    import asyncio
    from src.core.contracts.debug import DebugPayload

    debug = DebugPayload().model_dump(mode="json")
    assert debug["answer_provenance"] is None
    if omit_provenance:
        debug.pop("answer_provenance")
    payload = {"status": status, "response": None, "problem": problem,
               "message": message, "debug": debug}
    store = _session_store(payload)
    request = _request(store, query="정상 질문", session_id="no-answer-provenance", include_debug=True)
    service = AgentRequestService(runtime_cleaner=_FakeCleaner(), session_store=store)

    async def collect():
        return [event async for event in service.stream(request_id="no-answer", request_data=request)]

    events = asyncio.run(collect())
    final = next(event.data for event in events if event.event == "final_response")
    assert final["status"] == status
    assert final["debug"]["answer_provenance"] is None
    assert final["debug"]["observability_status"] == "ok"
    assert final["debug"]["missing_required_debug_fields"] == []
    store.close_all()


@pytest.mark.parametrize("status", ["completed", "partial"])
def test_answer_outcomes_still_require_provenance(status):
    import asyncio
    from src.core.contracts.debug import DebugPayload
    from tests.web.answer_fixtures import response_payload

    payload = {"status": status, "response": response_payload("검증된 답변"),
               "problem": _problem() if status == "partial" else None,
               "debug": DebugPayload().model_dump(mode="json")}
    store = _session_store(payload)
    request = _request(store, query="정상 질문", session_id="answer-provenance", include_debug=True)
    service = AgentRequestService(runtime_cleaner=_FakeCleaner(), session_store=store)

    async def collect():
        return [event async for event in service.stream(request_id="answer", request_data=request)]

    events = asyncio.run(collect())
    final = next(event.data for event in events if event.event == "final_response")
    assert final["status"] == status
    assert final["response"] == payload["response"]
    assert final["debug"]["observability_status"] == "failed"
    assert "answer_provenance" in final["debug"]["missing_required_debug_fields"]
    store.close_all()


def test_unexpected_service_failure_has_one_safe_terminal_and_keeps_manifest():
    store = _session_store({"response": None})
    service = AgentRequestService(runtime_cleaner=_FakeCleaner(), session_store=store)
    request = _request(store, query="정상 질문입니다", session_id="failure-session")
    with store.locked_session("failure-session") as (entry, _):
        def fail(*args, **kwargs):
            raise RuntimeError("PRIVATE provider traceback credential=secret")
        entry.agent.run_agent_flow = fail
    app = create_app()
    app.state.agent_request_service = service
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.post("/agent/stream", json=request.model_dump(mode="json"))
    finally:
        client.close()
    assert "PRIVATE" not in response.text
    assert "secret" not in response.text
    events = list(iter_sse_events([response.content]))
    assert [event.event for event in events] == ["request_started", "final_response", "done"]
    final = events[1].data
    assert final["status"] == "failed"
    assert final["response"] is None
    assert final["problem"]["code"] == "internal_error"
    assert final["message"] == final["problem"]["message"]
    assert final["upload_manifest"]["epoch"] == request.uploads.epoch
    assert final["request_id"] == events[0].data["request_id"]
    store.close_all()


def test_failure_before_session_does_not_invent_an_empty_attachment_snapshot():
    import asyncio

    store = _session_store({})
    request = _request(store, query="정상 질문", session_id="failure-before-session")

    class FailingCleaner:
        def run_once(self, **kwargs):
            raise RuntimeError("PRIVATE startup failure")

    service = AgentRequestService(runtime_cleaner=FailingCleaner(), session_store=store)

    async def collect():
        return [event async for event in service.stream(request_id="early-failure", request_data=request)]

    events = asyncio.run(collect())
    assert [event.event for event in events] == ["request_started", "final_response", "done"]
    final = events[1].data
    assert final["status"] == "failed"
    assert final["upload_manifest"] is None
    assert final["request_id"] == "early-failure"
    assert "PRIVATE" not in json.dumps(final)
    store.close_all()


@pytest.mark.parametrize("include_debug", [False, True])
def test_invalid_debug_does_not_replace_the_checked_result_or_attachment_snapshot(include_debug):
    import asyncio
    from tests.web.answer_fixtures import response_payload

    payload = response_payload("검증된 답변")
    store = _session_store({"response": payload, "debug": ["PRIVATE diagnostic data"]})
    request = _request(store, query="정상 질문", session_id="invalid-debug", include_debug=include_debug)
    service = AgentRequestService(runtime_cleaner=_FakeCleaner(), session_store=store)

    async def collect():
        return [event async for event in service.stream(request_id="debug-failure", request_data=request)]

    events = asyncio.run(collect())
    final = next(event.data for event in events if event.event == "final_response")
    assert final["status"] == "completed"
    assert final["response"] == payload
    assert final["upload_manifest"]["epoch"] == request.uploads.epoch
    assert "PRIVATE" not in json.dumps(final)
    if include_debug:
        assert final["debug"]["observability_status"] == "failed"
        assert "DEBUG_NORMALIZATION_FAILED" in final["debug"]["error_codes"]
    else:
        assert final["debug"] is None
    store.close_all()






def test_llm_diagnostic_metadata_survives_debug_normalization():
    import asyncio
    from src.core.llm_errors import LLMCallError, LLMDiagnostic, make_problem

    store = _session_store({})
    service = AgentRequestService(runtime_cleaner=_FakeCleaner(), session_store=store)
    request = _request(store, query="정상 질문", session_id="diagnostic-session", include_debug=True)
    diagnostic = LLMDiagnostic(
        code="provider_schema_invalid", stage="planner", model="test-model", schema_name="PlannerOutput",
        schema_hash="schema-fingerprint", provider_status=400, provider_code="invalid_json_schema",
        provider_request_id="provider-request", attempt=1,
    )
    with store.locked_session("diagnostic-session") as (entry, _):
        def fail(*args, **kwargs):
            raise LLMCallError(make_problem("provider_schema_invalid", "planner"), diagnostic)
        entry.agent.run_agent_flow = fail

    async def collect():
        return [event async for event in service.stream(request_id="diagnostic-request", request_data=request)]

    events = asyncio.run(collect())
    result = next(event.data for event in events if event.event == "final_response")
    assert result["status"] == "failed"
    assert result["problem"]["code"] == "provider_schema_invalid"
    assert result["debug"]["llm_diagnostics"] == [diagnostic.model_dump(mode="json")]
    assert result["request_id"] == "diagnostic-request"
    store.close_all()
