from concurrent.futures import ThreadPoolExecutor, TimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import pytest
from openai import APITimeoutError

from src.infra.chroma_store import build_openai_embeddings


@pytest.mark.parametrize("timeout_seconds", [None, 0.1])
def test_embedding_stalled_response_finishes_without_retry(monkeypatch, timeout_seconds):
    received = Event()
    release = Event()
    requests = []

    class StalledProvider(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(self.path)
            received.set()
            release.wait(35)
            self.close_connection = True

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), StalledProvider)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_BASE", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    provider = build_openai_embeddings("test-key", timeout_seconds=timeout_seconds)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(provider.embed_query, "bounded embedding request")
        assert received.wait(5), "provider did not receive the request"
        try:
            failure = future.exception(timeout=25 if timeout_seconds is None else 3)
        except TimeoutError:
            pytest.fail("embedding request remained blocked past the server timeout budget")
        assert isinstance(failure, APITimeoutError)
        assert requests == ["/v1/embeddings"]
    finally:
        release.set()
        pool.shutdown(wait=True)
        provider.client._client.close()
        server.shutdown()
        server.server_close()
        thread.join(5)
