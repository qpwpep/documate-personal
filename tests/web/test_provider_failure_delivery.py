"""Provider failure preserves the session through the real SDK, graph and HTTP client."""
from __future__ import annotations

import socket
from threading import Event, Thread

import httpx
import pytest
import uvicorn
from langchain_openai import ChatOpenAI

from src.app.client import AgentRequestContext, AgentSessionClient
from src.app.web.app import create_app
from src.infra.settings import AppSettings


@pytest.fixture
def agent_server(tmp_path, monkeypatch):
    """Serve the real app and runtime on loopback with isolated files and no external I/O."""
    settings = AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test",
        slack_bot_token="", slack_default_user_id="", slack_default_dm_email="",
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
        session = server_app.state.session_store.get_or_create(session_id)._ensure_session()
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
    assert final.result.request_id == events[0].data["request_id"]
    assert final.result.request_id
    assert final.data["debug"] is None
    assert len(provider_requests) == 1
    assert session.snapshot_conversation_memory() == before
    assert session.previous_response is None
    assert session.pending_action is None
    stages = [event.data["stage"] for event in events if event.event == "stage_started"]
    assert stages == ["planner"]
    for private in ("PRIVATE", "oneOf", "credential=secret", "private-provider-request"):
        assert private not in str(final.data)
