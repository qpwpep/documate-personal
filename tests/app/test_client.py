"""Stream failures must not replay questions or retain an unconfirmed session."""

from __future__ import annotations

import json

import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

from src.app.client import AgentRequestContext, AgentSessionClient
from src.core.answer_schema import finalize_answer, text_document
from src.core.uploads import UploadManifest


def frame(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


class RawStream:
    """Replace only bytes arriving from HTTP; requests still reads the stream."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.released = False

    def stream(self, *_args, **_kwargs):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        pass

    def release_conn(self):
        self.released = True


@pytest.fixture
def transport(monkeypatch):
    def install(chunks, *, status=200):
        sent = []
        response = requests.Response()
        response.status_code = status
        response.headers["Content-Type"] = "text/event-stream; charset=utf-8"
        response.raw = RawStream(chunks)
        if status in {307, 308}:
            response.headers["Location"] = "/forwarded-agent"

        def send(_adapter, request, **_kwargs):
            sent.append((request.method, request.url))
            assert len(sent) == 1, "A question must never be replayed automatically"
            response.request = request
            response.url = request.url
            return response

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
        return response, sent

    return install


@pytest.fixture
def client():
    return AgentSessionClient(
        AgentRequestContext(fastapi_url="http://localhost:8000", session_id="session-1"),
        manifest=UploadManifest(epoch="confirmed", revision=2, files=[]),
    )


@pytest.mark.parametrize("initial_event,code", [
    pytest.param(None, "empty_stream", id="empty-eof"),
    pytest.param("done", "missing_final_response", id="done-without-final"),
])
def test_stream_without_final_response_fails_and_invalidates_session(transport, client, initial_event, code):
    chunks = [frame(initial_event, {})] if initial_event else []
    response, sent = transport(chunks)

    events = list(client.stream("질문"))

    assert [event.event for event in events] == ([initial_event] if initial_event else []) + ["error"]
    assert events[-1].data["code"] == code
    assert all(event.result is None for event in events)
    assert client.manifest is None
    assert response.raw.released
    assert sent == [("POST", "http://localhost:8000/agent/stream")]


@pytest.mark.parametrize("progress_received,failure", [
    pytest.param(False, "disconnect", id="connection-error"),
    pytest.param(True, "disconnect", id="connection-interrupted"),
    pytest.param(True, "read-timeout", id="read-timeout"),
])
def test_interrupted_stream_fails_without_replaying_question(transport, client, progress_received, failure):
    error = (ReadTimeoutError(None, "/agent/stream", "read timed out") if failure == "read-timeout"
             else requests.exceptions.ConnectionError("connection lost"))
    chunks = [frame("request_started", {"request_id": "request-1"})] if progress_received else []
    response, sent = transport([*chunks, error])

    events = list(client.stream("결과를 저장해줘"))

    expected_code = ("timeout" if failure == "read-timeout"
                     else "connection_interrupted" if progress_received else "connection_error")
    assert [event.event for event in events] == (["request_started"] if progress_received else []) + ["error"]
    assert events[-1].data["code"] == expected_code
    assert all(event.result is None for event in events)
    assert client.manifest is None
    assert response.raw.released
    assert sent == [("POST", "http://localhost:8000/agent/stream")]


def test_redirect_does_not_forward_question_post(transport, client):
    _, sent = transport([b"redirected"], status=307)

    events = list(client.stream("결과를 저장해줘"))

    assert sent == [("POST", "http://localhost:8000/agent/stream")]
    assert [event.event for event in events] == ["error"]
    assert events[0].data["code"] == "http_error"
    assert events[0].data["status_code"] == 307
    assert events[0].result is None
    assert client.manifest is None


def test_session_accepts_utf8_characters_split_across_http_chunks(transport, client):
    expected = finalize_answer(text_document("한글 답변 🧪"), [])
    committed = UploadManifest(epoch="confirmed", revision=3, files=[])
    payload = {
        "response": expected.model_dump(mode="json"),
        "upload_manifest": committed.model_dump(mode="json"),
    }
    encoded = frame("final_response", payload)
    # One-byte chunks cross both multibyte characters and SSE frame boundaries.
    response, sent = transport([encoded[index:index + 1] for index in range(len(encoded))])

    events = list(client.stream("질문 🧪"))

    assert [event.event for event in events] == ["final_response"]
    assert events[0].data == payload
    assert events[0].result.response == expected
    assert events[0].result.upload_manifest == committed
    assert client.manifest == committed
    assert response.raw.released
    assert len(sent) == 1


def test_truncated_utf8_fails_and_invalidates_session(transport, client):
    incomplete = b'event: final_response\ndata: {"response": "' + "🧪".encode("utf-8")[:-1]
    response, sent = transport([frame("request_started", {"request_id": "request-utf8"}), incomplete])

    events = list(client.stream("질문"))

    assert [event.event for event in events] == ["request_started", "error"]
    assert events[-1].data["code"] == "invalid_stream"
    assert events[-1].observation.error_type == "stream_parse_error"
    assert all(event.result is None for event in events)
    assert client.manifest is None
    assert response.raw.released
    assert len(sent) == 1
