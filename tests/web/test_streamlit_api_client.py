from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import requests

from src.app.web.streamlit_api_client import (
    AgentCallResult, AgentRequestContext, _iter_sse_events, get_agent_response, stream_agent_response,
)
from src.core.answer_schema import export_answer_text
from tests.web.answer_fixtures import answer_response, cited_response


class StreamResponse:
    def __init__(self, frames, *, broken=False):
        self.status_code = 200
        self.frames = frames
        self.broken = broken

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_content(self, **_):
        yield from self.frames
        if self.broken:
            raise RuntimeError("stream broke")


def context():
    return AgentRequestContext(fastapi_url="http://localhost:8000", session_id="session-1")


def http_response(payload, status=200):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return response


def test_http_response_retains_document_citations_checks_and_actions():
    """HTTP preserves the complete final response, including source snapshots and actions."""
    expected = cited_response()
    with patch("requests.sessions.Session.request", return_value=http_response({"response": expected.model_dump(mode="json"), "trace": "t", "debug": None})):
        result = get_agent_response("질문", context())
    assert result == AgentCallResult(response=expected)


def test_request_preserves_session_and_upload_context():
    """The public HTTP request carries the selected session, upload and Slack target."""
    captured = []
    def request(_session, method, url, **kwargs):
        captured.append(kwargs["json"])
        return http_response({"response": answer_response().model_dump(mode="json")})

    request_context = AgentRequestContext(
        fastapi_url="http://localhost:8000", session_id="session-1",
        slack_user_id="U123", slack_email="test@example.com", slack_channel_id="C123",
        upload_file_path="uploads/session-1/code.py",
    )
    with patch("requests.sessions.Session.request", request):
        result = get_agent_response("질문", request_context)
    assert result.response == answer_response()
    assert captured == [{
        "query": "질문", "session_id": "session-1", "slack_user_id": "U123",
        "slack_email": "test@example.com", "slack_channel_id": "C123",
        "upload_file_path": "uploads/session-1/code.py",
    }]


@pytest.mark.parametrize("error, expected", [
    (requests.exceptions.Timeout(), "요청이 타임아웃되었습니다"),
    (requests.exceptions.ConnectionError(), "FastAPI 서버에 연결할 수 없습니다"),
    (RuntimeError("boom"), "boom"),
])
def test_transport_failure_returns_displayable_document(error, expected):
    """Transport failures remain readable through the same typed response path."""
    with patch("requests.sessions.Session.request", side_effect=error):
        result = get_agent_response("질문", context())
    assert expected in export_answer_text(result.response)
    assert result.response.citations == []


@pytest.mark.parametrize("payload", [
    {"response": "plain string"},
    {"response": {"answer": "old payload", "claims": [], "evidence": [], "confidence": None}},
])
def test_invalid_response_is_reported_without_partial_legacy_parsing(payload):
    """Malformed or obsolete payloads cannot silently lose structured information."""
    with patch("requests.sessions.Session.request", return_value=http_response(payload)):
        result = get_agent_response("질문", context())
    assert "오류" in export_answer_text(result.response)


def test_http_error_status_is_visible():
    """An unsuccessful HTTP response is rendered as a typed error document."""
    with patch("requests.sessions.Session.request", return_value=http_response({"error":"server exploded"}, 500)):
        result = get_agent_response("질문", context())
    assert "상태 코드 500" in export_answer_text(result.response)


def test_chunked_sse_restores_same_response_as_http():
    """Arbitrarily split SSE frames retain the complete response, not only body text."""
    expected = cited_response()
    frame = "event: final_response\ndata: " + json.dumps({"response":expected.model_dump(mode="json"), "trace":"t", "debug":None}, ensure_ascii=False) + "\n\n"
    events = list(_iter_sse_events([frame[:17], frame[17:67], frame[67:]]))
    assert len(events) == 1
    assert events[0].result == AgentCallResult(response=expected)


def test_stream_fallback_keeps_citations_checks_and_actions():
    """Fallback before the first event keeps the same public response fields."""
    expected = AgentCallResult(response=cited_response())
    with patch("src.app.web.streamlit_api_client.requests.post", side_effect=requests.exceptions.ConnectionError()), patch(
        "src.app.web.streamlit_api_client.get_agent_response", return_value=expected,
    ):
        events = list(stream_agent_response("질문", context()))
    assert len(events) == 1
    assert events[0].result == expected
    assert events[0].data["response"] == expected.response.model_dump(mode="json")


def test_stream_break_after_progress_reports_error_without_repeating_request():
    """A broken active stream produces an error rather than repeating the user action."""
    response = StreamResponse(['event: request_started\ndata: {"request_id":"r1"}\n\n'], broken=True)
    with patch("src.app.web.streamlit_api_client.requests.post", return_value=response):
        events = list(stream_agent_response("질문", context()))
    assert [event.event for event in events] == ["request_started", "error"]
    assert "stream broke" in events[-1].data["message"]
