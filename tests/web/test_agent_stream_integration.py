from __future__ import annotations

import socket
from pathlib import Path
from threading import Event, Thread

import pytest
import requests
import uvicorn

from src.app.web.app import create_app
from src.app.web.streamlit_api_client import AgentRequestContext, stream_agent_response
from src.core.answer_schema import export_answer_text
from src.core.conversation_memory import DEFAULT_QUERY_MAX_CHARS
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.judge_llm import LLMJudge
from src.eval.online_runner import _run_single_case
from src.infra.settings import AppSettings


@pytest.fixture
def agent_server(tmp_path, monkeypatch):
    """Serve the real app and runtime on loopback with isolated files and no external I/O."""
    settings = AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test",
        slack_bot_token="", slack_default_user_id="", slack_default_dm_email="",
    )
    monkeypatch.setattr("src.app.web.app.get_settings", lambda: settings)
    monkeypatch.setattr("src.infra.runtime_paths.get_uploads_dir", lambda: tmp_path / "uploads")
    monkeypatch.setattr("src.app.web.cleanup.get_uploads_dir", lambda: tmp_path / "uploads")
    monkeypatch.setattr("src.app.web.cleanup.get_save_text_output_dir", lambda: tmp_path / "save_text")
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


def _benchmark(endpoint: str, fixtures: Path, *, query: str, upload_fixture=None):
    return _run_single_case(
        run_id="sse-integration",
        endpoint=endpoint,
        fixtures_path=fixtures / "cases.jsonl",
        case=BenchmarkCase(
            case_id="stream-contract", category="docs_only", query=query,
            upload_fixture=upload_fixture,
        ),
        timeout_seconds=10,
        judge=LLMJudge(model_name="unused", enabled=False),
        config=BenchmarkConfig(judge_enabled=False),
    )


def test_streamlit_uses_the_real_sse_route_and_runtime(agent_server):
    """The UI client receives a complete answer through real HTTP, routing and session execution."""
    endpoint, app, _ = agent_server
    # The public reset command exercises the real manager without invoking an LLM or tool.
    events = list(stream_agent_response("exit", AgentRequestContext(fastapi_url=endpoint, session_id="ui-reset")))
    assert [event.event for event in events] == ["request_started", "final_response"]
    final = events[-1]
    assert export_answer_text(final.result.response) == "Chat session has been reset. Start again."
    assert final.data["response"] == final.result.response.model_dump(mode="json")
    assert "Session ID: ui-reset, Request ID:" in final.data["trace"]
    assert final.data["debug"] is None
    assert app.state.session_store.active_session_ids() == {"ui-reset"}


def test_benchmark_uses_the_real_sse_route_and_preserves_debug(agent_server):
    """The online runner receives response, trace and debug through the deployed HTTP contract."""
    endpoint, app, tmp_path = agent_server
    result = _benchmark(endpoint, tmp_path, query="exit")
    assert result.endpoint == endpoint + "/agent/stream"
    assert result.http_status == 200
    assert export_answer_text(result.response) == "Chat session has been reset. Start again."
    assert result.runtime_errors == result.response_errors == []
    assert result.request_id in result.trace
    assert result.debug["model_usage_status"] == "deterministic"
    assert result.debug["token_usage"]["total_tokens"] == 0
    assert result.debug["observed_hits"] == []
    assert result.latency_ms_e2e >= result.latency_ms_server
    assert app.state.session_store.active_session_ids() == {result.session_id}


def test_clients_handle_http_validation_errors_without_starting_a_session(agent_server):
    """Both migrated clients report HTTP rejection before any agent execution."""
    endpoint, app, tmp_path = agent_server
    query = "x" * (DEFAULT_QUERY_MAX_CHARS + 1)
    events = list(stream_agent_response(query, AgentRequestContext(fastapi_url=endpoint, session_id="invalid")))
    result = _benchmark(endpoint, tmp_path, query=query)
    assert [event.event for event in events] == ["error"]
    assert events[0].data["code"] == "http_error"
    assert events[0].data["status_code"] == 422
    assert result.http_status == 422
    assert result.runtime_errors[0].startswith("HTTP 422:")
    assert result.response is None
    assert app.state.session_store.active_session_ids() == set()


def test_clients_do_not_accept_http_200_when_the_service_emits_an_error(agent_server):
    """An SSE execution error and done cannot become a successful answer or trigger a second run."""
    endpoint, app, tmp_path = agent_server
    events = list(stream_agent_response("exit", AgentRequestContext(
        fastapi_url=endpoint, session_id="invalid-upload", upload_file_path="src/__init__.py",
    )))
    fixtures = tmp_path / "fixtures"
    (fixtures / "uploads").mkdir(parents=True)
    (fixtures / "uploads" / "unsupported.txt").write_text("unsupported", encoding="utf-8")
    result = _benchmark(endpoint, fixtures, query="exit", upload_fixture="unsupported.txt")
    assert [event.event for event in events] == ["request_started", "error", "done"]
    assert "UPLOAD_PATH_INVALID" in events[1].data["message"]
    assert result.http_status == 200
    assert result.response is None
    assert any("SSE error:" in error and "UPLOAD_PATH_INVALID" in error for error in result.runtime_errors)
    assert result.response_errors == ["SSE final_response missing (done received)"]
    assert not result.release_pass
    assert app.state.session_store.active_session_ids() == set()


def test_retired_route_is_unavailable_over_http(agent_server):
    """The server exposes no JSON execution route after the migration."""
    endpoint, app, _ = agent_server
    response = requests.post(endpoint + "/agent", json={"query": "exit", "session_id": "retired"}, timeout=5)
    assert response.status_code == 404
    assert app.state.session_store.active_session_ids() == set()
