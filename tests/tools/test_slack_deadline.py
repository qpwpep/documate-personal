"""Slack keeps one request deadline across lookup, DM creation and delivery."""
from __future__ import annotations

import time

import pytest
from slack_sdk.web import WebClient

from src.core.llm_errors import LLMCallError
from src.infra.llm_boundary import request_deadline
from src.infra.tools import slack_notify
from tests.tools.test_slack_notify import _delivery, slack_http_boundary


@pytest.mark.parametrize("expire_after, expected_paths", [
    (None, []),
    ("/users.lookupByEmail", ["/users.lookupByEmail"]),
    ("/conversations.open", ["/users.lookupByEmail", "/conversations.open"]),
])
def test_expired_request_prevents_the_next_slack_http_call(slack_http_boundary, expire_after, expected_paths):
    requests, responses = slack_http_boundary
    tool = slack_notify.build_slack_notify_tool("test-token")

    with request_deadline(10) as deadline:
        if expire_after is None:
            deadline.expires_at = 0
        else:
            def expire():
                deadline.expires_at = 0
                return ({"ok": True, "user": {"id": "UINTENDED"}}
                        if expire_after == "/users.lookupByEmail" else
                        {"ok": True, "channel": {"id": "DTARGET"}})
            responses[expire_after] = expire

        with pytest.raises(LLMCallError) as caught:
            tool(text="only within the request deadline", delivery=_delivery())

    assert caught.value.problem.code == "request_timeout"
    assert [request["path"] for request in requests] == expected_paths


def test_each_slack_http_timeout_uses_remaining_time_without_changing_the_shared_client(
    slack_http_boundary, monkeypatch,
):
    requests, responses = slack_http_boundary
    clients = []
    timeouts = []
    create_client = slack_notify.create_slack_client
    send_http = WebClient._perform_urllib_http_request_internal

    def capture_client(token):
        client = create_client(token)
        client.timeout = 20
        clients.append(client)
        return client

    def capture_timeout(self, url, req):
        timeouts.append(self.timeout)
        return send_http(self, url, req)

    monkeypatch.setattr(slack_notify, "create_slack_client", capture_client)
    monkeypatch.setattr(WebClient, "_perform_urllib_http_request_internal", capture_timeout)
    tool = slack_notify.build_slack_notify_tool("test-token")
    with request_deadline(15) as deadline:
        def after_lookup():
            deadline.expires_at = time.monotonic() + 10
            return {"ok": True, "user": {"id": "UINTENDED"}}

        def after_open():
            deadline.expires_at = time.monotonic() + 5
            return {"ok": True, "channel": {"id": "DTARGET"}}

        responses["/users.lookupByEmail"] = after_lookup
        responses["/conversations.open"] = after_open
        result = tool(text="bounded HTTP timeouts", delivery=_delivery())

    assert result.status == "sent"
    assert [request["path"] for request in requests] == [
        "/users.lookupByEmail", "/conversations.open", "/chat.postMessage",
    ]
    assert len(timeouts) == 3
    assert all(0 < timeout <= limit for timeout, limit in zip(timeouts, (15, 10, 5)))
    assert clients[0].timeout == 20
    responses.clear()
    tool(text="another request keeps its own timeout", delivery=_delivery())
    assert timeouts[3:] == [20, 20, 20]


def test_late_send_success_preserves_its_receipt(slack_http_boundary):
    requests, responses = slack_http_boundary
    tool = slack_notify.build_slack_notify_tool("test-token")

    with request_deadline(10) as deadline:
        def late_success():
            deadline.expires_at = 0
            return {"ok": True, "channel": "DTARGET", "ts": "42.0"}

        responses["/chat.postMessage"] = late_success
        result = tool(text="a confirmed send must not be lost", delivery=_delivery())

    assert result.status == "sent"
    assert result.target.channel_id == "DTARGET"
    assert result.message_ts == "42.0"
    assert len([request for request in requests if request["path"] == "/chat.postMessage"]) == 1
