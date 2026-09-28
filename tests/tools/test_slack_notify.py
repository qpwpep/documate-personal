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


@pytest.mark.parametrize("kind,value,channel_id", [
    ("email", "intended@example.invalid", "DTARGET"),
    ("user", "UINTENDED", "DTARGET"),
    ("user", "WINTENDED", "DTARGET"),
    ("channel", "C123", "C123"),
])
def test_explicit_recipient_sends_to_the_confirmed_target_once(slack_http_boundary, kind, value, channel_id):
    requests, responses = slack_http_boundary
    responses["/chat.postMessage"] = {"ok": True, "channel": channel_id, "ts": "123.4"}
    original = _delivery(kind, value)
    tool = build_slack_notify_tool("test-token")

    result = tool(text="intended message", delivery=original)
    repeated = tool(text="intended message", delivery=result)

    assert _sent_requests(requests) == [{"path": "/chat.postMessage", "payload": {"channel": channel_id, "text": "intended message"}}]
    assert result.status == "sent"
    assert result.target.channel_id == channel_id
    assert result.message_ts == "123.4"
    assert result.intent == original.intent
    assert result.selection == original.selection
    assert repeated == result


def test_preselected_default_preserves_its_source(slack_http_boundary):
    requests, _responses = slack_http_boundary
    original = _delivery("user", "UDEFAULT", default=True)

    result = build_slack_notify_tool("test-token")(text="message", delivery=original)

    assert len(_sent_requests(requests)) == 1
    assert result.status == "sent"
    assert result.intent.state == "omitted"
    assert result.selection.source == "configured_default"
    assert result.target.user_id == "UDEFAULT"


def test_open_dm_failure_retains_the_resolved_user_for_retry(slack_http_boundary):
    requests, responses = slack_http_boundary
    responses["/conversations.open"] = {"ok": False, "error": "missing_scope"}
    tool = build_slack_notify_tool("test-token")

    first = tool(text="message", delivery=_delivery())

    assert first.status == "not_sent"
    assert first.failure.stage == "open_dm"
    assert first.resolved_user_id == "UINTENDED"
    assert _sent_requests(requests) == []

    responses["/users.lookupByEmail"] = {"ok": True, "user": {"id": "UOTHER"}}
    responses["/conversations.open"] = {"ok": True, "channel": {"id": "DTARGET"}}
    second = tool(text="message", delivery=first)

    assert second.status == "sent"
    assert second.selection == first.selection
    assert second.target.user_id == "UINTENDED"
    assert {request["payload"]["users"] for request in requests if request["path"] == "/conversations.open"} == {"UINTENDED"}
    assert len(_sent_requests(requests)) == 1


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


@pytest.mark.parametrize("endpoint,slack_error,error_code", [
    ("/users.lookupByEmail", "access_denied", "permission_denied"),
    ("/users.lookupByEmail", "accesslimited", "permission_denied"),
    ("/users.lookupByEmail", "team_access_not_granted", "permission_denied"),
    ("/users.lookupByEmail", "enterprise_is_restricted", "permission_denied"),
    ("/users.lookupByEmail", "not_allowed_token_type", "authentication_failed"),
    ("/users.lookupByEmail", "two_factor_setup_required", "authentication_failed"),
    ("/chat.postMessage", "access_denied", "permission_denied"),
    ("/chat.postMessage", "accesslimited", "permission_denied"),
    ("/chat.postMessage", "team_access_not_granted", "permission_denied"),
    ("/chat.postMessage", "enterprise_is_restricted", "permission_denied"),
    ("/chat.postMessage", "not_allowed_token_type", "authentication_failed"),
    ("/chat.postMessage", "two_factor_setup_required", "authentication_failed"),
    ("/chat.postMessage", "not_in_channel", "permission_denied"),
    ("/chat.postMessage", "app_access_restricted", "permission_denied"),
    ("/chat.postMessage", "restricted_action_read_only_channel", "permission_denied"),
    ("/chat.postMessage", "restricted_action_thread_only_channel", "permission_denied"),
])
def test_access_rejections_report_configuration_repair_without_changing_recipient(slack_http_boundary, endpoint, slack_error, error_code):
    requests, responses = slack_http_boundary
    responses[endpoint] = {"ok": False, "error": slack_error}
    original = _delivery()

    result = build_slack_notify_tool("test-token")(text="message", delivery=original)

    assert result.status == "not_sent"
    assert result.selection == original.selection
    assert result.intent == original.intent
    assert result.failure.code == error_code
    assert result.failure.slack_error == slack_error
    assert result.failure.next_action == "fix_configuration"
    assert len(_sent_requests(requests)) == (1 if endpoint == "/chat.postMessage" else 0)


@pytest.mark.parametrize("response", [None, (503, {"ok": False, "error": "internal_error"}, {})])
def test_temporary_lookup_failure_is_definitely_not_sent(slack_http_boundary, response):
    requests, responses = slack_http_boundary
    responses["/users.lookupByEmail"] = response

    result = build_slack_notify_tool("test-token")(text="message", delivery=_delivery())

    assert result.status == "not_sent"
    assert result.failure.code == "temporary_failure"
    assert result.failure.next_action == "retry_same_target"
    assert _sent_requests(requests) == []


@pytest.mark.parametrize("endpoint", ["/users.lookupByEmail", "/chat.postMessage"])
def test_rate_limit_preserves_retry_after_without_automatic_retry(slack_http_boundary, endpoint):
    requests, responses = slack_http_boundary
    responses[endpoint] = (429, {"ok": False, "error": "ratelimited"}, {"Retry-After": "17"})

    result = build_slack_notify_tool("test-token")(text="message", delivery=_delivery())

    assert result.status == "not_sent"
    assert result.failure.code == "rate_limited"
    assert result.failure.retry_after_seconds == 17
    assert len([request for request in requests if request["path"] == endpoint]) == 1
    assert len(_sent_requests(requests)) == (1 if endpoint == "/chat.postMessage" else 0)


@pytest.mark.parametrize("response", [
    None,
    (503, {"ok": False, "error": "internal_error"}, {}),
    (503, {"ok": False, "error": "missing_scope"}, {}),
    {"ok": True, "channel": "DOTHER", "ts": "1.0"},
    {"ok": True, "channel": "DTARGET"},
])
def test_unconfirmed_delivery_never_automatically_resends(slack_http_boundary, response):
    requests, responses = slack_http_boundary
    responses["/chat.postMessage"] = response
    tool = build_slack_notify_tool("test-token")

    first = tool(text="message", delivery=_delivery())
    repeated = tool(text="message", delivery=first)

    assert first.status == "unknown"
    assert first.failure.next_action == "verify_delivery"
    assert first.target.channel_id == "DTARGET"
    assert repeated == first
    assert len(_sent_requests(requests)) == 1


@pytest.mark.parametrize("endpoint,response", [
    ("/users.lookupByEmail", {"ok": True, "user": {}}),
    ("/users.lookupByEmail", {"ok": True, "user": {"id": "not-a-user"}}),
    ("/conversations.open", {"ok": True, "channel": {"id": "CWRONG"}}),
])
def test_malformed_resolution_never_reaches_message_delivery(slack_http_boundary, endpoint, response):
    requests, responses = slack_http_boundary
    responses[endpoint] = response

    result = build_slack_notify_tool("test-token")(text="message", delivery=_delivery())

    assert result.status == "not_sent"
    assert result.failure.code == "protocol_error"
    assert _sent_requests(requests) == []


def test_missing_auth_preserves_the_selected_recipient(slack_http_boundary):
    requests, _responses = slack_http_boundary
    original = _delivery()

    result = build_slack_notify_tool(None)(text="message", delivery=original)

    assert result.status == "not_sent"
    assert result.failure.code == "authentication_failed"
    assert result.selection == original.selection
    assert requests == []


def test_previously_blocked_intent_is_not_erased_by_the_sender(slack_http_boundary):
    requests, _responses = slack_http_boundary
    delivery = SlackDelivery(
        intent=UnresolvedRecipient(raw_input="our team", reason="ambiguous"), status="not_sent",
        failure=SlackFailure(stage="selection", code="recipient_ambiguous", message="Please select one recipient.", next_action="correct_input"),
    )

    result = build_slack_notify_tool(None)(text="message", delivery=delivery)

    assert result == delivery
    assert requests == []
