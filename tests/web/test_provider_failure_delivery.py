"""Provider failure preserves the session through the real SDK, graph and HTTP client."""
from __future__ import annotations

import json
import socket
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import httpx
import pytest
import uvicorn
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.app.client import AgentRequestContext, AgentSessionClient
from src.app.web.app import create_app
from src.core.answer_schema import text_document
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.settings import AppSettings


@pytest.fixture
def agent_server(tmp_path, monkeypatch):
    """Serve the real app and runtime on loopback with isolated files and no external I/O."""
    settings = AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test",
        slack_bot_token="", slack_default_user_id="", slack_default_dm_email="",
        chat_model="gpt-4o-mini", planner_model="gpt-4o-mini", summary_model="gpt-4o-mini",
        synthesis_use_responses_api=False,
    )
    monkeypatch.setattr("src.app.web.app.get_settings", lambda: settings)
    monkeypatch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: tmp_path)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    connect = socket.socket.connect

    def loopback_only(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1"}:
            raise AssertionError(f"External network is forbidden in this integration test: {address[0]}")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    ready = Event()

    class TestServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            ready.set()

    app = create_app()
    server = TestServer(uvicorn.Config(
        app, host="127.0.0.1", port=0, log_config=None, access_log=False,
        loop="asyncio", http="h11", ws="none",
    ))
    errors = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        endpoint = f"http://127.0.0.1:{listener.getsockname()[1]}"

        def serve():
            try:
                server.run(sockets=[listener])
            except BaseException as exc:
                errors.append(exc)
            finally:
                ready.set()

        thread = Thread(target=serve, name="agent-integration-server", daemon=True)
        thread.start()
        try:
            assert ready.wait(10), "local server did not start"
            assert not errors, errors
            assert server.started
            yield endpoint, app, tmp_path
        finally:
            server.should_exit = True
            thread.join(10)
            assert not thread.is_alive(), "local server did not shut down"
            assert not errors, errors



def test_provider_schema_rejection_preserves_session_without_committing_an_answer(agent_server, monkeypatch):
    endpoint, server_app, _ = agent_server
    provider_requests = []

    def provider(request):
        provider_requests.append(request)
        return httpx.Response(400, json={"error": {
            "message": "PRIVATE invalid response_format oneOf; credential=secret",
            "type": "invalid_request_error", "param": "response_format", "code": "invalid_json_schema",
        }}, headers={"x-request-id": "private-provider-request"})

    with httpx.Client(transport=httpx.MockTransport(provider)) as provider_client:
        monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **kwargs: ChatOpenAI(
            **kwargs, http_client=provider_client, base_url="https://provider.test/v1",
        ))
        session_id = "schema-failure-delivery"
        client = AgentSessionClient(AgentRequestContext(fastapi_url=endpoint, session_id=session_id))
        manifest = client.refresh_uploads()
        session = server_app.state.session_store.get_or_create(session_id).session
        before = session.snapshot_conversation_memory()

        events = list(client.stream("pandas merge 사용법을 공식 문서 기준으로 설명해줘"))
    finals = [event for event in events if event.event == "final_response"]
    assert len(finals) == 1
    assert not [event for event in events if event.event == "error"]
    final = finals[0]
    assert final.result.status == "failed"
    assert final.result.problem.code == "provider_schema_invalid"
    assert final.result.problem.next_action == "fix_configuration"
    assert final.result.response is None
    assert final.result.upload_manifest == manifest
    assert final.result.execution_id == events[0].data["execution_id"]
    assert final.result.execution_id == final.observation.execution_id
    assert len(final.result.execution_id) == 36
    assert final.data["debug"] is None
    assert len(provider_requests) == 1
    assert session.snapshot_conversation_memory() == before
    assert session.previous_response is None
    assert session.pending_request is None
    stages = [event.data["stage"] for event in events if event.event == "stage_started"]
    assert stages == ["planner"]
    for private in ("PRIVATE", "oneOf", "credential=secret", "private-provider-request"):
        assert private not in str(final.data)


@pytest.mark.parametrize("waiting_for", ["validation", "session_lock"])
def test_stream_stays_connected_through_validation_and_session_lock(agent_server, monkeypatch, waiting_for):
    endpoint, app, _ = agent_server
    release = Event()
    validation_entered = Event()
    lock_acquired = Event()

    def model_response(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        schema = payload["response_format"]["json_schema"]["schema"]
        content = (InitialPlannerOutput(
            use_retrieval=False, tasks=[],
            request_contract=WireRequestContract(slack_recipient={"state": "omitted"}),
        ) if "use_retrieval" in schema["properties"] else text_document("대기 후 받은 답변입니다."))
        return httpx.Response(200, json={
            "id": "chat-stream-lifecycle", "object": "chat.completion", "created": 1,
            "model": payload["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content.model_dump_json()}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        })

    def decisions_response(request: httpx.Request) -> httpx.Response:
        validation_entered.set()
        if waiting_for == "validation":
            assert release.wait(15), "validation was never released by the receiving client"
        payload = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "gpt-6-luna", "answers": [
                {"type": "predicate", "name": question["name"], "probability": 0.99}
                for question in payload["questions"]
            ], "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32},
        })

    def hold_session_lock() -> None:
        with app.state.session_store.locked_session("waiting-session"):
            lock_acquired.set()
            assert release.wait(15), "session lock was never released by the receiving client"

    with (
        httpx.Client(transport=httpx.MockTransport(model_response)) as model_http,
        httpx.Client(transport=httpx.MockTransport(decisions_response)) as decisions_http,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **kwargs: ChatOpenAI(
            **kwargs, http_client=model_http, base_url="https://provider.test/v1",
        ))
        monkeypatch.setattr("src.infra.llm.OpenAI", lambda **kwargs: OpenAI(
            **kwargs, http_client=decisions_http, base_url="https://provider.test/v1",
        ))
        context = AgentRequestContext(fastapi_url=endpoint, session_id="waiting-session", timeout_seconds=2)
        client = AgentSessionClient(context)
        manifest = client.refresh_uploads()
        holder = pool.submit(hold_session_lock) if waiting_for == "session_lock" else None
        events = []
        heartbeats = []
        try:
            if holder is not None:
                assert lock_acquired.wait(5)
            for event in client.stream("짧은 답변을 작성해줘"):
                events.append(event)
                if event.event == "heartbeat":
                    heartbeats.append(event)
                    if waiting_for == "validation":
                        assert validation_entered.is_set()
                    else:
                        assert not validation_entered.is_set()
                        assert not any(item.event == "stage_started" for item in events)
                    # The real client must survive longer than its read timeout.
                    if event.data["elapsed_ms"] > context.timeout_seconds * 1000:
                        release.set()
            if waiting_for == "validation":
                assert validation_entered.is_set(), [(event.event, event.data) for event in events]
            assert release.is_set(), "no keepalive arrived while execution was blocked"
        finally:
            release.set()
            if holder is not None:
                holder.result(timeout=5)

    assert heartbeats
    assert not [event for event in events if event.event == "error"]
    final = next(event for event in events if event.event == "final_response")
    assert final.result.status == "completed"
    assert final.result.upload_manifest == manifest
    assert all(event.data["execution_id"] == final.result.execution_id for event in heartbeats)
    assert final.result.response.content.blocks[0].content[0].text == "대기 후 받은 답변입니다."
