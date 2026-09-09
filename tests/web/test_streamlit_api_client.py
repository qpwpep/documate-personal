from __future__ import annotations

import json

import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

from src.app.web.streamlit_api_client import (
    AgentCallResult,
    AgentRequestContext,
    _iter_sse_events,
    stream_agent_response,
)
from tests.web.answer_fixtures import answer_response, cited_response


class StreamResponse:
    def __init__(self, frames=(), *, status=200, text="", content_type="text/event-stream"):
        self.status_code = status
        self.headers = {"Content-Type": content_type} if content_type else {}
        self.text = text
        self.frames = frames
        self.closed = False
        self.frames_read = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True
        return False

    def iter_content(self, **_):
        for frame in self.frames:
            self.frames_read += 1
            if isinstance(frame, Exception):
                raise frame
            yield frame


@pytest.fixture
def transport(monkeypatch):
    calls = []

    def install(response):
        def request(_session, method, url, **kwargs):
            calls.append({"method": method, "url": url, **kwargs})
            if isinstance(response, Exception):
                raise response
            return response

        monkeypatch.setattr(requests.sessions.Session, "request", request)
        return calls

    return install


def context():
    return AgentRequestContext(fastapi_url="http://localhost:8000", session_id="session-1")


def frame(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def test_stream_preserves_complete_final_response_and_stops_without_done(transport):
    """A valid final response preserves all fields and completes without another read."""
    expected = cited_response()
    payload = {
        "response": expected.model_dump(mode="json"),
        "trace": "planner -> synthesis",
        "debug": {"metrics": {"total_tokens": 42}, "retrieval": {"query": "질문"}},
    }
    response = StreamResponse([
        frame("final_response", payload),
        requests.exceptions.ConnectionError("must not read after final response"),
    ])
    calls = transport(response)

    events = list(stream_agent_response("질문", context()))

    assert len(events) == 1
    assert events[0].event == "final_response"
    assert events[0].data == payload
    assert events[0].result == AgentCallResult(response=expected)
    assert response.frames_read == 1
    assert response.closed
    assert len(calls) == 1
    assert calls[0]["url"] == "http://localhost:8000/agent/stream"


def test_request_preserves_session_upload_and_slack_context(transport):
    """The single streamed POST carries the selected session, upload and Slack target."""
    calls = transport(StreamResponse([
        frame("final_response", {"response": answer_response().model_dump(mode="json")}),
    ]))
    request_context = AgentRequestContext(
        fastapi_url="http://localhost:8000/", session_id="session-1",
        slack_user_id="U123", slack_email="test@example.com", slack_channel_id="C123",
        upload_file_path="uploads/session-1/code.py",
    )

    events = list(stream_agent_response("질문", request_context))

    assert events[0].result.response == answer_response()
    assert len(calls) == 1
    assert calls[0]["method"] == "post"
    assert calls[0]["url"] == "http://localhost:8000/agent/stream"
    assert calls[0]["stream"] is True
    assert calls[0]["headers"]["Accept"] == "text/event-stream"
    assert calls[0]["json"] == {
        "query": "질문", "session_id": "session-1", "slack_user_id": "U123",
        "slack_email": "test@example.com", "slack_channel_id": "C123",
        "upload_file_path": "uploads/session-1/code.py",
    }


@pytest.mark.parametrize("error, code, message", [
    (requests.exceptions.Timeout("timeout"), "timeout", "시간이 초과"),
    (requests.exceptions.ConnectionError("disconnected"), "connection_error", "첫 이벤트"),
    (RuntimeError("boom"), "stream_error", "boom"),
])
def test_failure_before_first_event_reports_error_without_repeating_request(transport, error, code, message):
    """An initial failure is visible without a retry that could repeat server actions."""
    calls = transport(error)

    events = list(stream_agent_response("질문", context()))

    assert len(events) == 1
    assert events[0].event == "error"
    assert events[0].result is None
    assert events[0].data["code"] == code
    assert message in events[0].data["message"]
    assert len(calls) == 1


@pytest.mark.parametrize("progress_received", [False, True])
def test_stream_read_timeout_remains_timeout_without_repeating_request(transport, progress_received):
    """A read timeout wrapped by requests stays a timeout before or after progress."""
    class RawStream:
        closed = False

        def stream(self, *_args, **_kwargs):
            if progress_received:
                yield frame("request_started", {"request_id": "r1"}).encode("utf-8")
            raise ReadTimeoutError(None, "/agent/stream", "read timed out")

        def close(self):
            self.closed = True

        def release_conn(self):
            pass

    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "text/event-stream; charset=utf-8"
    response.raw = RawStream()
    calls = transport(response)

    events = list(stream_agent_response("질문", context()))

    assert [event.event for event in events] == (["request_started"] if progress_received else []) + ["error"]
    assert events[-1].data["code"] == "timeout"
    assert "시간이 초과" in events[-1].data["message"]
    assert response.raw.closed
    assert len(calls) == 1


def test_http_error_is_visible_without_reading_sse_or_repeating_request(transport):
    """A failed HTTP status exposes its status and body without another request."""
    response = StreamResponse(status=503, text="temporarily unavailable")
    calls = transport(response)

    events = list(stream_agent_response("질문", context()))

    assert len(events) == 1
    assert events[0].event == "error"
    assert events[0].data["code"] == "http_error"
    assert events[0].data["status_code"] == 503
    assert "503" in events[0].data["message"]
    assert "temporarily unavailable" in events[0].data["message"]
    assert response.frames_read == 0
    assert response.closed
    assert len(calls) == 1


@pytest.mark.parametrize("status", [307, 308])
def test_redirect_is_reported_without_forwarding_post(monkeypatch, status):
    """The real requests redirect machinery cannot replay a streamed agent POST."""
    sent = []

    def send(_adapter, request, **_kwargs):
        sent.append((request.method, request.url))
        response = requests.Response()
        response.request = request
        response.url = request.url
        response._content_consumed = True
        if len(sent) == 1:
            response.status_code = status
            response.headers["Location"] = "/forwarded-agent"
            response._content = b"redirected"
        else:
            response.status_code = 200
            response.headers["Content-Type"] = "text/event-stream"
            response._content = frame("final_response", {
                "response": answer_response().model_dump(mode="json"),
            }).encode("utf-8")
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)

    events = list(stream_agent_response("질문", context()))

    assert sent == [("POST", "http://localhost:8000/agent/stream")]
    assert [event.event for event in events] == ["error"]
    assert events[0].data["code"] == "http_error"
    assert events[0].data["status_code"] == status


@pytest.mark.parametrize("content_type", ["application/json", "text/html", None])
def test_non_sse_response_is_rejected_without_reading_body(transport, content_type):
    """Stale JSON, proxy pages and missing media types cannot masquerade as SSE."""
    response = StreamResponse([json.dumps({"response": "stale response"})], content_type=content_type)
    calls = transport(response)

    events = list(stream_agent_response("질문", context()))

    assert [event.event for event in events] == ["error"]
    assert events[0].data["code"] == "invalid_stream"
    assert "Content-Type" in events[0].data["message"]
    assert response.frames_read == 0
    assert response.closed
    assert len(calls) == 1


def test_sse_content_type_accepts_charset_parameter(transport):
    """A standard SSE media type with its UTF-8 parameter preserves the final answer."""
    transport(StreamResponse([
        frame("final_response", {"response": answer_response().model_dump(mode="json")}),
    ], content_type="text/event-stream; charset=utf-8"))

    events = list(stream_agent_response("질문", context()))

    assert [event.event for event in events] == ["final_response"]
    assert events[0].result.response == answer_response()


@pytest.mark.parametrize("frames", [[], [": heartbeat\n\n"]])
def test_empty_stream_reports_error_without_repeating_request(transport, frames):
    """A stream with no data events is a visible empty-stream failure."""
    calls = transport(StreamResponse(frames))

    events = list(stream_agent_response("질문", context()))

    assert [event.event for event in events] == ["error"]
    assert events[0].data["code"] == "empty_stream"
    assert "비어" in events[0].data["message"]
    assert len(calls) == 1


@pytest.mark.parametrize("event", ["request_started", "done"])
def test_stream_without_final_response_is_not_success(transport, event):
    """Neither HTTP 200 nor a done event replaces the required final response."""
    calls = transport(StreamResponse([frame(event, {})]))

    events = list(stream_agent_response("질문", context()))

    assert [item.event for item in events] == [event, "error"]
    assert events[-1].data["code"] == "missing_final_response"
    assert "최종 응답" in events[-1].data["message"]
    assert all(item.result is None for item in events)
    assert len(calls) == 1


@pytest.mark.parametrize("error", [
    requests.exceptions.ConnectionError("stream broke"),
    requests.exceptions.ChunkedEncodingError("stream broke"),
])
def test_stream_break_after_progress_reports_interruption_without_repeating_request(transport, error):
    """A broken active stream reports an interruption without repeating user actions."""
    calls = transport(StreamResponse([
        frame("request_started", {"request_id": "r1"}),
        error,
    ]))

    events = list(stream_agent_response("질문", context()))

    assert [event.event for event in events] == ["request_started", "error"]
    assert events[-1].data["code"] == "connection_interrupted"
    assert "도중" in events[-1].data["message"]
    assert "stream broke" in events[-1].data["message"]
    assert len(calls) == 1


def test_server_error_without_final_response_keeps_original_error(transport):
    """An SSE error remains visible when the server ends without a final response."""
    error = {"message": "업로드 경로를 확인해 주세요.", "request_id": "r1"}
    transport(StreamResponse([frame("error", error), frame("done", {})]))

    events = list(stream_agent_response("질문", context()))

    assert [event.event for event in events] == ["error", "done"]
    assert events[0].data == error
    assert all(event.result is None for event in events)


def test_server_error_can_be_followed_by_complete_final_response(transport):
    """A server error does not discard the subsequent response and diagnostic payload."""
    payload = {
        "response": answer_response("요청 처리 중 오류가 발생했습니다.").model_dump(mode="json"),
        "trace": "planner",
        "debug": {"runtime_error": "failed"},
    }
    transport(StreamResponse([
        frame("error", {"message": "failed"}),
        frame("final_response", payload),
    ]))

    events = list(stream_agent_response("질문", context()))

    assert [event.event for event in events] == ["error", "final_response"]
    assert events[-1].data == payload
    assert events[-1].result.response.model_dump(mode="json") == payload["response"]


@pytest.mark.parametrize("payload", [
    {"response": "plain string"},
    {"response": {"answer": "old payload", "claims": [], "evidence": [], "confidence": None}},
])
def test_invalid_final_response_is_reported_without_partial_parsing(transport, payload):
    """Malformed final responses cannot silently lose structured answer information."""
    transport(StreamResponse([frame("final_response", payload)]))

    events = list(stream_agent_response("질문", context()))

    assert [event.event for event in events] == ["error"]
    assert events[0].data["code"] == "invalid_stream"
    assert events[0].result is None


@pytest.mark.parametrize("encoded", [
    "event: request_started\ndata: not-json\n\n",
    "event: request_started\ndata: []\n\n",
    "event: request_started\ndata: {}\n",
])
def test_invalid_or_incomplete_stream_reports_error_without_repeating_request(transport, encoded):
    """Malformed or truncated SSE frames produce a visible failure without a retry."""
    calls = transport(StreamResponse([encoded]))

    events = list(stream_agent_response("질문", context()))

    assert [event.event for event in events] == ["error"]
    assert events[0].data["code"] == "invalid_stream"
    assert events[0].result is None
    assert len(calls) == 1


def test_chunked_sse_preserves_response_trace_and_debug():
    """Arbitrarily split frames preserve the complete event and typed answer."""
    expected = cited_response()
    payload = {"response": expected.model_dump(mode="json"), "trace": "t", "debug": None}
    encoded = frame("final_response", payload)

    events = list(_iter_sse_events([encoded[:17], encoded[17:67], encoded[67:]]))

    assert len(events) == 1
    assert events[0].data == payload
    assert events[0].result == AgentCallResult(response=expected)
