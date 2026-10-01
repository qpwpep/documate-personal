from __future__ import annotations

import json

import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

from src.app.client import (
    AgentCallResult,
    AgentRequestContext,
    stream_agent_response,
)
from src.core.uploads import UploadContext, UploadManifest, UploadSyncRequest
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




























@pytest.mark.parametrize("manifest", [[], {"epoch": "epoch", "revision": -1, "files": []}])
def test_invalid_final_manifest_does_not_confirm_a_partial_response(transport, manifest):
    """An invalid attachment snapshot rejects the final response without confirming stale state."""
    calls = transport(StreamResponse([frame("final_response", {
        "response": answer_response().model_dump(mode="json"),
        "upload_manifest": manifest,
    })]))

    events = list(stream_agent_response("질문", context(), uploads=UploadContext(epoch="e", revision=0)))

    assert [event.event for event in events] == ["error"]
    assert events[0].data["code"] == "invalid_stream"
    assert events[0].result is None
    assert len(calls) == 1
































