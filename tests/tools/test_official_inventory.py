"""Official symbol inventories locate pages without guessing their filenames."""

import json
import zlib

import pytest
import requests

from src.core.planner_schema import RetrievalRequirement
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool


def response(url, body, *, status=200):
    result = requests.Response()
    result.url = url
    result.status_code = status
    result._content = body.encode() if isinstance(body, str) else body
    result._content_consumed = True
    result.headers["Content-Type"] = "application/octet-stream" if url.endswith("objects.inv") else "text/html; charset=utf-8"
    result.encoding = "utf-8"
    return result


def inventory(entries):
    header = b"# Sphinx inventory version 2\n# Project: Example\n# Version: current\n# The remainder of this file is compressed using zlib.\n"
    return header + zlib.compress("\n".join(entries).encode())


@pytest.mark.parametrize("library,version,symbol,root,name,location", [
    ("Python", None, "asyncio.wait_for", "https://docs.python.org/3/", "asyncio.wait_for", "library/asyncio-task.html#$"),
    ("Python", "3.10", "builtins.zip", "https://docs.python.org/3.10/", "zip", "library/functions.html#$"),
    ("Python", None, "json.dumps", "https://docs.python.org/3/", "json.dumps", "library/json.html#$"),
    ("Python", None, "str.removeprefix", "https://docs.python.org/3/", "str.removeprefix", "builtins/stdtypes.html#$"),
    ("pytorch", None, "torch.nn.Module.eval", "https://docs.pytorch.org/docs/stable/", "torch.nn.Module.eval", "generated/torch.nn.Module.html#$"),
])
def test_inventory_locates_requested_page_when_search_misses_it(monkeypatch, library, version, symbol, root, name, location):
    target = root + location.split("#")[0]
    body = f"<title>{name} official documentation</title><main><dl><dt>{name}()¶</dt><dd><p>Requested API behavior.</p></dd></dl></main>"
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if url == root + "objects.inv":
            return response(url, inventory([f"{name} std:label -1 unrelated.html -", f"{name} py:function 1 {location} -"]))
        return response(url, body if url == target else "Not found", status=200 if url == target else 404)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": []})))
    search = build_docs_search_tool(AppSettings(_env_file=None, openai_api_key="test", tavily_api_key="test"))

    result = search(symbol, requirement=RetrievalRequirement(library=library, version=version, symbols=[symbol]), k=1)

    assert result.hits
    assert result.hits[0].evidence.snapshot.source_uri == target
    assert result.hits[0].evidence.snapshot.parser == "official_html"
    assert "Requested API behavior." in result.hits[0].evidence.excerpt
    assert calls == [root + "objects.inv", target]


@pytest.mark.parametrize("location", [
    "https://attacker.example/api.html#$",
    "https://numpy.org/doc/stable/api.html#$",
    "../outside.html#$",
])
def test_inventory_location_stays_inside_the_requested_documentation_scope(monkeypatch, location):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return response(url, inventory([f"json.dumps py:function 1 {location} -"]))

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": []})))
    search = build_docs_search_tool(AppSettings(_env_file=None, openai_api_key="test", tavily_api_key="test"))

    result = search("json.dumps", requirement=RetrievalRequirement(library="Python", symbols=["json.dumps"]))

    assert result.hits == []
    assert calls == ["https://docs.python.org/3/objects.inv"]


@pytest.mark.parametrize("payload", [b"not an inventory", inventory(["x" * (4 * 1024 * 1024 + 1)])],
                         ids=["invalid_header", "oversized_decompressed"])
def test_invalid_or_oversized_inventory_preserves_original_search_evidence(monkeypatch, payload):
    archived = "https://docs.pytorch.org/docs/2.9/generated/torch.no_grad.html"
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": [{
        "url": archived, "title": "torch.no_grad 2.9", "content": "Archived no_grad explanation.", "score": 0.9,
    }]})))
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: response(url, payload))
    search = build_docs_search_tool(AppSettings(_env_file=None, openai_api_key="test", tavily_api_key="test"))

    result = search("torch.no_grad", requirement=RetrievalRequirement(library="pytorch", symbols=["torch.no_grad"]))

    assert result.hits
    assert all(hit.evidence.snapshot.source_uri == archived for hit in result.hits)
    assert all(hit.evidence.snapshot.parser == "tavily" for hit in result.hits)
    assert "official_inventory_unavailable" in result.diagnostics.warnings


def test_full_page_refresh_does_not_retain_an_unrelated_fragment(monkeypatch):
    base = "https://pandas.pydata.org/docs/reference/api/pandas.merge.html"
    source = {"url": base + "#unrelated", "title": "merge", "content": "Cached description.", "score": 0.9}
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": [source]})))
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: response(url, "<main><h1>pandas.merge</h1><p>Merge tables.</p></main>"))
    search = build_docs_search_tool(AppSettings(_env_file=None, openai_api_key="test", tavily_api_key="test"))
    requirement = RetrievalRequirement(library="pandas")

    first = search("merge", requirement=requirement)
    source["url"] = base + "#pandas.merge"
    second = search("merge", requirement=requirement, refresh=True)

    assert first.hits[0].evidence.snapshot.source_uri == base
    assert first.hits[0].evidence.snapshot == second.hits[0].evidence.snapshot


def test_stable_html_redirect_captures_the_actual_release_page(monkeypatch):
    stable = "https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html"
    resolved = "https://docs.pytorch.org/docs/2.15/generated/torch.no_grad.html"
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if url.endswith("objects.inv"):
            return response(url, inventory(["torch.no_grad py:class 1 generated/torch.no_grad.html#$ -"]))
        if url == stable:
            return response(url, '<title>Redirecting</title><meta http-equiv="refresh" content="0; url=../../2.15/generated/torch.no_grad.html">')
        return response(url, "<title>torch.no_grad — PyTorch 2.15</title><main><h1>torch.no_grad</h1><p>Disable gradient calculation.</p></main>")

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": []})))
    search = build_docs_search_tool(AppSettings(_env_file=None, openai_api_key="test", tavily_api_key="test"))

    result = search("torch.no_grad", requirement=RetrievalRequirement(library="pytorch", symbols=["torch.no_grad"]))

    assert result.hits
    evidence = result.hits[0].evidence
    assert evidence.snapshot.source_uri == stable
    assert evidence.snapshot.parser_config["resolved_url"] == resolved
    assert evidence.snapshot.title == "torch.no_grad — PyTorch 2.15"
    assert "Disable gradient calculation." in evidence.excerpt
    assert calls == ["https://docs.pytorch.org/docs/stable/objects.inv", stable, resolved]


@pytest.mark.parametrize("target", ["https://attacker.example/docs", "https://numpy.org/doc/stable/api.html"])
def test_html_redirect_cannot_leave_the_requested_library(monkeypatch, target):
    stable = "https://pandas.pydata.org/docs/reference/api/pandas.merge.html"
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return response(url, f'<meta http-equiv="refresh" content="0; url={target}">')

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": []})))
    search = build_docs_search_tool(AppSettings(_env_file=None, openai_api_key="test", tavily_api_key="test"))

    result = search("pandas.merge", requirement=RetrievalRequirement(library="pandas", symbols=["pandas.merge"]))

    assert result.hits == []
    assert calls == [stable]
    assert "official_page_refresh_failed" in result.diagnostics.warnings


def test_http_and_html_redirects_share_the_existing_limit(monkeypatch):
    stable = "https://pandas.pydata.org/docs/reference/api/pandas.merge.html"
    alternate = "https://pandas.pydata.org/docs/reference/api/pandas.other.html"
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if url == stable:
            result = response(url, "", status=302)
            result.headers["Location"] = alternate
            return result
        return response(url, f'<meta http-equiv="refresh" content="0; url={stable}">')

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": []})))
    search = build_docs_search_tool(AppSettings(_env_file=None, openai_api_key="test", tavily_api_key="test"))

    result = search("pandas.merge", requirement=RetrievalRequirement(library="pandas", symbols=["pandas.merge"]))

    assert result.hits == []
    assert calls == [stable, alternate, stable]
    assert "official_page_refresh_failed" in result.diagnostics.warnings
