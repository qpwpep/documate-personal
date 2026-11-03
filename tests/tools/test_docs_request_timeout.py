"""A request deadline bounds provider search and official source capture together."""

import json
from types import SimpleNamespace

import pytest
import requests

from src.core.llm_errors import LLMCallError
from src.core.planner_schema import RetrievalRequirement
from src.infra import llm_boundary
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool


PAGE_URL = "https://pandas.pydata.org/docs/reference/api/pandas.merge.html"
PAGE = "<main><h1>pandas.merge</h1><p>Merge tables using database joins.</p></main>"


def response(url, body, *, status=200):
    result = requests.Response()
    result.url = url
    result.status_code = status
    result._content = body.encode()
    result._content_consumed = True
    result.headers["Content-Type"] = "text/html; charset=utf-8"
    result.encoding = "utf-8"
    return result


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(llm_boundary, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


@pytest.fixture
def search():
    return build_docs_search_tool(AppSettings(
        _env_file=None, openai_api_key="test", tavily_api_key="test", docs_search_timeout_seconds=5,
    ))


def test_expired_request_does_not_start_provider_search(search, clock, monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return response(url, '{"results": []}')

    monkeypatch.setattr(requests, "post", post)
    with llm_boundary.request_deadline(1):
        clock.now = 1
        with pytest.raises(LLMCallError) as caught:
            search("numpy", requirement=RetrievalRequirement(library="numpy", version="1.26"))

    assert caught.value.problem.code == "request_timeout"
    assert calls == []


@pytest.mark.parametrize("outcome", ["success", "http_error", "timeout", "connection_error"])
def test_late_provider_result_keeps_request_timeout_cause(search, clock, monkeypatch, outcome):
    timeouts = []

    def post(url, *, timeout, **kwargs):
        timeouts.append(timeout)
        clock.now = 2
        if outcome == "timeout":
            raise requests.Timeout("HTTP read deadline")
        if outcome == "connection_error":
            raise requests.ConnectionError("Connection lost")
        return response(url, '{"results": []}', status=429 if outcome == "http_error" else 200)

    monkeypatch.setattr(requests, "post", post)
    with llm_boundary.request_deadline(2):
        clock.now = 0.5
        with pytest.raises(LLMCallError) as caught:
            search("numpy", requirement=RetrievalRequirement(library="numpy", version="1.26"))

    assert caught.value.problem.code == "request_timeout"
    assert caught.value.diagnostic.budget_reason == "request_timeout"
    assert timeouts == [1.5]


def test_provider_and_redirect_use_remaining_request_time(search, clock, monkeypatch):
    requested = []

    def post(url, *, timeout, **kwargs):
        requested.append((url, timeout))
        clock.now = 1
        return response(url, '{"results": []}')

    def get(url, *, timeout, **kwargs):
        requested.append((url, timeout))
        if url == PAGE_URL:
            clock.now = 2
            result = response(url, "", status=302)
            result.headers["Location"] = PAGE_URL + "?redirected=1"
            return result
        clock.now = 3
        return response(url, PAGE)

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)
    with llm_boundary.request_deadline(4):
        result = search("pandas.merge", requirement=RetrievalRequirement(library="pandas", symbols=["pandas.merge"]))

    assert result.hits
    assert result.hits[0].evidence.snapshot.parser == "official_html"
    assert [timeout for _, timeout in requested] == [4, 3, 2]
    assert [url for url, _ in requested[1:]] == [PAGE_URL, PAGE_URL + "?redirected=1"]


@pytest.mark.parametrize("outcome", ["redirect", "chunk", "empty_stream", "timeout"])
def test_expired_official_capture_stops_without_cached_fallback(search, clock, monkeypatch, outcome):
    cached = {"url": PAGE_URL, "title": "pandas.merge", "content": "Old cached merge explanation.", "score": 0.9}
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": [cached]})))
    fetched = []

    def get(url, *, timeout, **kwargs):
        fetched.append((url, timeout))
        result = response(url, PAGE)
        if outcome == "redirect":
            clock.now = 2
            result.status_code = 302
            result.headers["Location"] = PAGE_URL + "?redirected=1"
        elif outcome == "timeout":
            clock.now = 2
            raise requests.Timeout("HTTP read deadline")
        else:
            def chunks(chunk_size):
                clock.now = 2
                if outcome == "chunk":
                    yield PAGE.encode()

            result.iter_content = chunks
        return result

    monkeypatch.setattr(requests, "get", get)
    with llm_boundary.request_deadline(2):
        clock.now = 0.5
        with pytest.raises(LLMCallError) as caught:
            search("pandas.merge", requirement=RetrievalRequirement(library="pandas", symbols=["pandas.merge"]))

    assert caught.value.problem.code == "request_timeout"
    assert fetched == [(PAGE_URL, 1.5)]


def test_expired_inventory_does_not_fall_back_to_other_pages(search, clock, monkeypatch):
    cached = {"url": "https://docs.python.org/3/library/json.html", "title": "json", "content": "JSON encoding.", "score": 0.9}
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": [cached]})))
    fetched = []

    def get(url, *, timeout, **kwargs):
        fetched.append((url, timeout))
        clock.now = 2
        raise requests.Timeout("Inventory read deadline")

    monkeypatch.setattr(requests, "get", get)
    with llm_boundary.request_deadline(2):
        with pytest.raises(LLMCallError) as caught:
            search("json.dumps", requirement=RetrievalRequirement(library="Python", symbols=["json.dumps"]))

    assert caught.value.problem.code == "request_timeout"
    assert fetched == [("https://docs.python.org/3/objects.inv", 2)]
