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
from src.core.uploads import UploadManifest
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


@pytest.mark.parametrize("content_type", ["text/event-stream", "text/event-stream; charset=utf-8"])
def test_stream_preserves_complete_final_response_and_stops_without_done(transport, content_type):
    """Both standard SSE media types preserve all fields and finish without another read."""
    expected = cited_response()
    manifest = UploadManifest(epoch="reset-epoch", revision=0, files=[])
    payload = {
        "response": expected.model_dump(mode="json"),
        "trace": "planner -> synthesis",
        "debug": {"metrics": {"total_tokens": 42}, "retrieval": {"query": "질문"}},
        "upload_manifest": manifest.model_dump(mode="json"),
    }
    response = StreamResponse([
        frame("final_response", payload),
        requests.exceptions.ConnectionError("must not read after final response"),
    ], content_type=content_type)
    calls = transport(response)

    events = list(stream_agent_response("질문", context()))

    assert len(events) == 1
    assert events[0].event == "final_response"
    assert events[0].data == payload
    assert events[0].result == AgentCallResult(response=expected, upload_manifest=manifest)
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
    (ReadTimeoutError(None, "/agent/stream", "read timed out"), "timeout", "시간이 초과"),
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


@pytest.mark.parametrize("manifest", [[], {"epoch": "epoch", "revision": -1, "files": []}])
def test_invalid_final_manifest_does_not_confirm_a_partial_response(transport, manifest):
    """An invalid attachment snapshot rejects the final response without confirming stale state."""
    calls = transport(StreamResponse([frame("final_response", {
        "response": answer_response().model_dump(mode="json"),
        "upload_manifest": manifest,
    })]))

    events = list(stream_agent_response("질문", context()))

    assert [event.event for event in events] == ["error"]
    assert events[0].data["code"] == "invalid_stream"
    assert events[0].result is None
    assert len(calls) == 1


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


def test_question_sends_confirmed_upload_revision_without_legacy_path(transport):
    """A question pins the confirmed attachment generation even when it has no new files."""
    from src.core.uploads import UploadContext
    calls = transport(StreamResponse([frame("final_response", {"response": answer_response().model_dump(mode="json")})]))
    request_context = AgentRequestContext(fastapi_url="http://localhost:8000", session_id="session-1", uploads=UploadContext(epoch="epoch-one", revision=2))
    events = list(stream_agent_response("두 파일을 비교해줘", request_context))
    assert events[0].result.response == answer_response()
    assert calls[0]["json"] == {"query": "두 파일을 비교해줘", "session_id": "session-1", "uploads": {"epoch": "epoch-one", "revision": 2}}


def test_upload_sync_failure_preserves_file_errors_and_does_not_retry(transport):
    """A failed attachment batch exposes per-file errors without silently replaying it."""
    from src.app.web.streamlit_api_client import UploadAPIError, sync_uploads
    response = requests.Response()
    response.status_code = 400
    detail = {"code": "UPLOAD_INVALID", "message": "첨부 실패", "files": [{"name": "bad.py", "code": "INVALID_UTF8", "message": "UTF-8 오류"}]}
    response._content = json.dumps({"detail": detail}).encode()
    calls = transport(response)
    with pytest.raises(UploadAPIError) as raised:
        sync_uploads("http://localhost:8000", "session-1", {"operation_id": "operation-1"})
    assert raised.value.status_code == 400
    assert raised.value.code == "UPLOAD_INVALID"
    assert raised.value.files == detail["files"]
    assert len(calls) == 1


def test_upload_sync_returns_server_confirmed_manifest(transport):
    """The UI receives the authoritative committed set after a successful sync."""
    from src.app.web.streamlit_api_client import sync_uploads
    response = requests.Response()
    response.status_code = 200
    expected = {"epoch": "epoch-one", "revision": 3, "files": []}
    response._content = json.dumps({"manifest": expected, "changed": True, "unchanged_names": []}).encode()
    calls = transport(response)
    result = sync_uploads("http://localhost:8000/", "session-1", {"operation_id": "operation-1"})
    assert result.manifest.model_dump(mode="json") == expected
    assert result.changed is True
    assert calls[0]["url"] == "http://localhost:8000/sessions/session-1/uploads/sync"
    assert calls[0]["allow_redirects"] is False


def test_diagnostic_request_preserves_transport_observation_and_server_error(transport):
    """Opt-in diagnostics retain server failures and the final response's HTTP timing."""
    from src.app.client import AgentRequestContext, stream_agent_response

    final = {"response": answer_response().model_dump(mode="json"), "debug": {"runtime_error": "failed"}}
    calls = transport(StreamResponse([
        frame("request_started", {"request_id": "request-one"}),
        frame("error", {"message": "failed"}),
        frame("final_response", final),
    ]))

    events = list(stream_agent_response("question", AgentRequestContext(
        fastapi_url="http://localhost:8000", session_id="session-one",
        include_debug=True, timeout_seconds=17,
    )))

    assert len(calls) == 1
    assert calls[0]["json"]["include_debug"] is True
    assert calls[0]["timeout"] == 17
    assert events[1].observation.error_source == "server"
    assert events[-1].data == final
    assert events[-1].observation.http_status == 200
    assert events[-1].observation.request_id == "request-one"
    assert events[-1].observation.final_received is True
    assert events[-1].observation.elapsed_ms >= 0


def test_transport_request_id_prefers_http_header_over_event_values(transport):
    """The HTTP request identity is preserved when an event supplies a different identifier."""
    response = StreamResponse([
        frame("request_started", {"request_id": "event-request"}),
        frame("final_response", {"response": answer_response().model_dump(mode="json")}),
    ])
    response.headers["x-request-id"] = "header-request"
    transport(response)

    events = list(stream_agent_response("question", context()))

    assert [event.observation.request_id for event in events] == ["header-request", "header-request"]


def test_session_uses_final_manifest_for_followup_and_refreshes_after_lost_response(monkeypatch):
    """A session follows server epochs and recovers by reading state without replaying questions."""
    from src.app.client import AgentRequestContext, AgentSessionClient

    initial = UploadManifest(epoch="initial", revision=2, files=[])
    reset = UploadManifest(epoch="reset", revision=0, files=[])
    recovered = UploadManifest(epoch="recovered", revision=1, files=[])
    calls = []
    responses = iter([
        StreamResponse([frame("final_response", {
            "response": answer_response().model_dump(mode="json"),
            "upload_manifest": reset.model_dump(mode="json"),
        })]),
        requests.exceptions.Timeout("lost response"),
        recovered,
        StreamResponse([frame("final_response", {
            "response": answer_response().model_dump(mode="json"),
            "upload_manifest": recovered.model_dump(mode="json"),
        })]),
    ])

    def send(_session, method, url, **kwargs):
        calls.append({"method": method, "payload": kwargs.get("json")})
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, UploadManifest):
            reply = requests.Response()
            reply.status_code = 200
            reply._content = response.model_dump_json().encode()
            return reply
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", send)
    client = AgentSessionClient(AgentRequestContext(
        fastapi_url="http://localhost:8000", session_id="same-session",
    ), manifest=initial)

    assert list(client.stream("exit"))[-1].result is not None
    assert client.manifest == reset
    assert list(client.stream("lost question"))[-1].data["code"] == "timeout"
    assert client.manifest is None
    assert list(client.stream("new question"))[-1].result is not None
    assert client.manifest == recovered
    assert [call["method"] for call in calls] == ["post", "post", "get", "post"]
    assert [call["payload"]["query"] for call in calls if call["method"] == "post"] == [
        "exit", "lost question", "new question",
    ]
    assert [call["payload"]["uploads"] for call in calls if call["method"] == "post"] == [
        manifest.context().model_dump() for manifest in [initial, reset, recovered]
    ]
    assert {call["payload"]["session_id"] for call in calls if call["method"] == "post"} == {"same-session"}


def test_invalid_final_manifest_keeps_diagnostics_but_invalidates_session(transport):
    """A rejected final manifest remains an identifiable schema error with its raw diagnostics."""
    from src.app.client import AgentRequestContext, AgentSessionClient

    payload = {"response": answer_response().model_dump(mode="json"),
               "upload_manifest": {"epoch": "invalid", "revision": -1}, "debug": {"trace": "saved"}}
    transport(StreamResponse([frame("final_response", payload)]))
    client = AgentSessionClient(AgentRequestContext(
        fastapi_url="http://localhost:8000", session_id="session-one",
    ), manifest=UploadManifest(epoch="confirmed", revision=1))

    events = list(client.stream("question"))

    assert client.manifest is None
    assert [event.event for event in events] == ["error"]
    assert events[0].result is None
    assert events[0].data["raw_final_response"] == payload
    assert events[0].observation.error_source == "client"
    assert events[0].observation.error_type == "agent_schema_error"
    assert events[0].observation.final_received is True


def test_unconfirmed_session_does_not_send_question_when_manifest_refresh_fails(transport):
    """An unavailable attachment confirmation endpoint prevents execution against guessed state."""
    from src.app.client import AgentRequestContext, AgentSessionClient, UploadAPIError

    calls = transport(requests.exceptions.ConnectionError("unavailable"))
    client = AgentSessionClient(AgentRequestContext(
        fastapi_url="http://localhost:8000", session_id="session-one",
    ))

    with pytest.raises(UploadAPIError):
        list(client.stream("question"))

    assert client.manifest is None
    assert [(call["method"], call["url"]) for call in calls] == [
        ("get", "http://localhost:8000/sessions/session-one/uploads"),
    ]


def test_uncertain_upload_sync_invalidates_confirmation_without_replaying(transport):
    """Losing a mutation response prevents later questions from reusing the pre-mutation revision."""
    from src.app.client import AgentRequestContext, AgentSessionClient, UploadAPIError
    from src.app.uploads import PendingUploadOperation

    calls = transport(requests.exceptions.Timeout("lost sync response"))
    client = AgentSessionClient(AgentRequestContext(
        fastapi_url="http://localhost:8000", session_id="session-one",
    ), manifest=UploadManifest(epoch="one", revision=2))
    operation = PendingUploadOperation(epoch="one", expected_revision=2, clear=True)

    with pytest.raises(UploadAPIError):
        client.sync_uploads(operation)

    assert client.manifest is None
    assert len(calls) == 1
    assert calls[0]["json"] == operation.request_payload()


def test_session_stages_syncs_and_queries_using_the_server_confirmed_uploads(monkeypatch, tmp_path):
    """File bytes pass through shared staging and sync before a question pins the committed revision."""
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace

    from src.app.client import AgentRequestContext, AgentSessionClient
    from src.app.uploads import PendingUploadOperation, discard_staged_files
    from src.core.uploads import UploadFileInfo

    uploaded_bytes = b"print('shared path')\n"
    initial = UploadManifest(epoch="one", revision=0)
    committed = UploadManifest(epoch="one", revision=1, files=[UploadFileInfo(
        file_id="file-one", name="code.py", size_bytes=len(uploaded_bytes),
        content_hash="sha256:" + hashlib.sha256(uploaded_bytes).hexdigest(),
        source_uri="upload://session-one/file-one",
    )])
    calls = []

    def send(_session, method, url, **kwargs):
        payload = kwargs.get("json")
        calls.append({"method": method, "url": url, "payload": payload})
        response = requests.Response()
        response.status_code = 200
        response._content_consumed = True
        if url.endswith("/uploads"):
            response._content = initial.model_dump_json().encode()
        elif url.endswith("/uploads/sync"):
            assert [(item["name"], Path(item["path"]).read_bytes()) for item in payload["add"]] == [
                ("code.py", uploaded_bytes),
            ]
            response._content = json.dumps({"manifest": committed.model_dump(), "changed": True}).encode()
        else:
            response.headers["Content-Type"] = "text/event-stream"
            response._content = frame("final_response", {
                "response": answer_response().model_dump(mode="json"),
                "upload_manifest": committed.model_dump(mode="json"),
            }).encode()
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", send)
    client = AgentSessionClient(AgentRequestContext(
        fastapi_url="http://localhost:8000", session_id="session-one",
    ))
    staged = client.stage_files([
        SimpleNamespace(name="code.py", getbuffer=lambda: uploaded_bytes),
    ], tmp_path, max_files=5, max_file_mib=1, max_total_mib=5)
    assert staged.errors == []
    assert client.manifest == initial
    operation = PendingUploadOperation(
        epoch=initial.epoch, expected_revision=initial.revision, files=staged.files,
    )

    assert client.sync_uploads(operation).manifest == committed
    assert list(client.stream("explain code.py"))[-1].result is not None

    assert client.manifest == committed
    assert [call["method"] for call in calls] == ["get", "post", "post"]
    assert calls[1]["payload"] == operation.request_payload()
    assert calls[2]["payload"] == {
        "query": "explain code.py", "session_id": "session-one",
        "uploads": committed.context().model_dump(),
    }
    discard_staged_files(staged.files, tmp_path)
    assert not list(tmp_path.rglob("*.py"))
