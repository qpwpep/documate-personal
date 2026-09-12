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
















def test_sse_content_type_accepts_charset_parameter(transport):
    """A standard SSE media type with its UTF-8 parameter preserves the final answer."""
    transport(StreamResponse([
        frame("final_response", {"response": answer_response().model_dump(mode="json")}),
    ], content_type="text/event-stream; charset=utf-8"))

    events = list(stream_agent_response("질문", context()))

    assert [event.event for event in events] == ["final_response"]
    assert events[0].result.response == answer_response()














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




def test_chunked_sse_preserves_response_trace_and_debug():
    """Arbitrarily split frames preserve the complete event and typed answer."""
    expected = cited_response()
    payload = {"response": expected.model_dump(mode="json"), "trace": "t", "debug": None}
    encoded = frame("final_response", payload)

    events = list(_iter_sse_events([encoded[:17], encoded[17:67], encoded[67:]]))

    assert len(events) == 1
    assert events[0].data == payload
    assert events[0].result == AgentCallResult(response=expected)






