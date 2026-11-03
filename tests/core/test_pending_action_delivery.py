from __future__ import annotations

import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import pytest
from slack_sdk.web import WebClient

from src.app.agent_manager import AgentFlowManager
from src.core.request_contracts import RequestContract
from src.infra.settings import AppSettings
from src.infra.tools.save_text import build_save_text_tool
from src.infra.tools.slack_notify import build_slack_notify_tool


def _contract(*, save="not_requested", slack="not_requested", **updates):
    return RequestContract.model_validate({
        "actions": {
            "save_text": {"intent": save, "evidence_ids": ["request"]},
            "slack_notify": {"intent": slack, "evidence_ids": ["request"]},
        },
        "evidence": [{"id": "request", "turn_id": "current", "quote": "사용자 요청", "scope": "current_request", "interpretation": "instruction"}],
        **updates,
    })


@pytest.fixture
def delivery_tools(tmp_path: Path, monkeypatch, request):
    """실제 저장 도구와 Slack SDK를 임시 파일 및 로컬 HTTP 경계에 연결한다."""
    messages = []
    configured_responses = getattr(request, "param", {})
    responses = iter(configured_responses) if isinstance(configured_responses, list) else None
    endpoint_responses = {
        path: iter(items) if isinstance(items, list) else iter([items])
        for path, items in configured_responses.items()
    } if isinstance(configured_responses, dict) else {}

    class SlackHandler(BaseHTTPRequestHandler):
        def _respond(self):
            parsed = urlsplit(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if body and "application/json" in self.headers.get("Content-Type", ""):
                payload = json.loads(body)
            else:
                values = parse_qs(body.decode() if body else parsed.query)
                payload = {key: items[0] for key, items in values.items()}
            messages.append({"path": parsed.path, "payload": payload})
            default = {
                "/users.lookupByEmail": {"ok": True, "user": {"id": "UINTENDED"}},
                "/conversations.open": {"ok": True, "channel": {"id": "D" + payload.get("users", "UTARGET")[1:]}},
                "/chat.postMessage": {"ok": True, "channel": payload.get("channel"), "ts": "1.0"},
            }[parsed.path]
            iterator = endpoint_responses.get(parsed.path, responses)
            response = next(iterator, default) if iterator is not None else default
            if response is None:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_POST = _respond
        do_GET = _respond

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlackHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    client = WebClient(token="test-token", base_url=f"http://127.0.0.1:{server.server_port}/", retry_handlers=[])
    monkeypatch.setattr("src.infra.tools.slack_notify.create_slack_client", lambda _token: client)
    output = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    settings = AppSettings(_env_file=None, openai_api_key="test-key", tavily_api_key="test-key", slack_bot_token="test-token")
    try:
        yield build_save_text_tool(), build_slack_notify_tool(settings.slack_bot_token), messages, output
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _manager(graph):
    return AgentFlowManager(
        settings=AppSettings(_env_file=None, openai_api_key="test-key", tavily_api_key="test-key"),
        graph=graph,
    )
