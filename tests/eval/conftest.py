from urllib.parse import urlsplit

import pytest
import requests

from src.core.uploads import UploadManifest


@pytest.fixture
def empty_upload_manifest_http(monkeypatch):
    """Confirm empty attachments for focused runner tests at the HTTP boundary.

    Tests opt in explicitly and supply their own streamed POST response. Any
    other HTTP request fails immediately instead of reaching a running service.
    """
    calls = []

    def request(_session, method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        path = urlsplit(url).path
        assert method.lower() == "get" and path.startswith("/sessions/") and path.endswith("/uploads"), (
            f"Unconfigured HTTP request in a focused evaluation test: {method} {url}"
        )
        response = requests.Response()
        response.status_code = 200
        response.headers["Content-Type"] = "application/json"
        response._content = UploadManifest(epoch="fixture-epoch", revision=0).model_dump_json().encode("utf-8")
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", request)
    return calls
