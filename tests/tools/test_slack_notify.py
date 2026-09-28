from __future__ import annotations

import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import pytest

from src.core.slack_contract import (
    ExplicitRecipient, OmittedRecipient, RecipientSelection, RecipientSelector, SlackDelivery,
    SlackFailure, UnresolvedRecipient,
)
from src.infra.settings import AppSettings
from src.infra.slack_utils import create_slack_client
from src.infra.tools.slack_notify import build_slack_notify_tool


@pytest.fixture
def slack_http_boundary(monkeypatch):
    """Use the real Slack SDK with an observable HTTP boundary."""
    requests = []
    responses = {}

    class SlackHandler(BaseHTTPRequestHandler):
        def _respond(self):
            parsed = urlsplit(self.path)
            raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if raw_body and "application/json" in self.headers.get("Content-Type", ""):
                payload = json.loads(raw_body)
            else:
                values = parse_qs(raw_body.decode() if raw_body else parsed.query)
                payload = {key: items[0] for key, items in values.items()}
            requests.append({"path": parsed.path, "payload": payload})
            response = responses.get(parsed.path, {
                "/users.lookupByEmail": {"ok": True, "user": {"id": "UINTENDED"}},
                "/conversations.open": {"ok": True, "channel": {"id": "DTARGET"}},
                "/chat.postMessage": {"ok": True, "channel": "DTARGET", "ts": "1.0"},
            }.get(parsed.path, {"ok": False, "error": "unknown_method"}))
            if response is None:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            status, headers = 200, {}
            if isinstance(response, tuple):
                status, response, headers = response
            body = json.dumps(response).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        do_GET = _respond
        do_POST = _respond

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlackHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    def local_client(token):
        client = create_slack_client(token)
        if client is not None:
            client.base_url = f"http://127.0.0.1:{server.server_port}/"
            client.timeout = 2
        return client

    monkeypatch.setattr("src.infra.tools.slack_notify.create_slack_client", local_client)
    try:
        yield requests, responses
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _delivery(kind="email", value="intended@example.invalid", *, default=False):
    selector = RecipientSelector(kind=kind, value=value)
    return SlackDelivery(
        intent=OmittedRecipient() if default else ExplicitRecipient(selector=selector),
        selection=RecipientSelection(
            request_id="test-request", source="configured_default" if default else "request_input", selector=selector,
        ),
    )


def _sent_requests(requests):
    return [request for request in requests if request["path"] == "/chat.postMessage"]


@pytest.mark.parametrize("slack_error,error_code", [
    ("users_not_found", "target_not_found"), ("missing_scope", "permission_denied"),
    ("invalid_auth", "authentication_failed"), ("ratelimited", "rate_limited"),
])
def test_explicit_lookup_failure_never_sends_to_configured_default(slack_http_boundary, slack_error, error_code):
    requests, responses = slack_http_boundary
    responses["/users.lookupByEmail"] = {"ok": False, "error": slack_error}
    settings = AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test-key",
        slack_bot_token="test-token", slack_default_user_id="UDEFAULT", slack_default_dm_email=None,
    )

    # A default exists in application configuration; the sender only receives the
    # explicit choice, so it has no capability to select that default on failure.
    original = _delivery()
    result = build_slack_notify_tool(settings.slack_bot_token)(text="intended message", delivery=original)

    assert _sent_requests(requests) == []
    assert result.status == "not_sent"
    assert result.selection == original.selection
    assert result.intent == original.intent
    assert result.failure.code == error_code
    assert result.failure.slack_error == slack_error
    assert result.failure.stage == "lookup"








def test_rejected_send_retries_the_already_resolved_channel(slack_http_boundary):
    requests, responses = slack_http_boundary
    responses["/chat.postMessage"] = {"ok": False, "error": "not_in_channel"}
    tool = build_slack_notify_tool("test-token")

    first = tool(text="message", delivery=_delivery())
    assert first.status == "not_sent"
    assert first.failure.code == "permission_denied"
    responses["/users.lookupByEmail"] = {"ok": True, "user": {"id": "UOTHER"}}
    responses["/conversations.open"] = {"ok": True, "channel": {"id": "DOTHER"}}
    responses["/chat.postMessage"] = {"ok": True, "channel": "DTARGET", "ts": "2.0"}

    second = tool(text="message", delivery=first)

    assert second.status == "sent"
    assert second.target == first.target
    assert {request["payload"]["channel"] for request in _sent_requests(requests)} == {"DTARGET"}
    assert len(_sent_requests(requests)) == 2














