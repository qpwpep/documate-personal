"""Stable documentation is captured from its current HTTP page, not search cache."""

import hashlib
import json

import pytest
import requests

from src.core.planner_schema import RetrievalRequirement
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool


URL = "https://pandas.pydata.org/docs/reference/api/pandas.merge.html"
PAGE = """<!doctype html><html><head><title>pandas.merge — pandas 3.1.0 documentation</title></head>
<body><nav>Archived 1.5 navigation</nav><main><article>
<h1>pandas.merge</h1><p>Merge current tables.</p>
<pre><code>result = merge(\n    left, right, on="key"\n)</code></pre>
<pre>def helper():\n    value = 1\n\n\n    return value</pre>
<dl><dt><strong>left</strong><span class="classifier">DataFrame</span></dt></dl>
<table><tr><td>5</td><td>6</td></tr></table>
<script>do_not_include()</script><aside>Unrelated links</aside>
</article></main><footer>Site footer</footer></body></html>"""


def response(url, text, status=200, content_type="text/html; charset=utf-8"):
    result = requests.Response()
    result.url = url
    result.status_code = status
    result._content = text.encode("utf-8")
    result._content_consumed = True
    result.headers["Content-Type"] = content_type
    result.encoding = "utf-8"
    return result


@pytest.fixture
def search(monkeypatch):
    stale = {"url": URL, "title": "pandas.merge — pandas 3.0.6 documentation",
             "content": "Obsolete cached description.", "score": 0.9}
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(
        url, json.dumps({"results": [stale]}), content_type="application/json"))
    return build_docs_search_tool(AppSettings(_env_file=None, openai_api_key="test", tavily_api_key="test"))


def test_stable_search_uses_current_title_body_and_replaces_previous_cache(search, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: response(url, PAGE))
    result = search("pandas merge usage", requirement=RetrievalRequirement(library="pandas"), k=1)

    assert result.hits
    captured = result.hits[0].evidence
    assert captured.snapshot.title == "pandas.merge — pandas 3.1.0 documentation"
    assert captured.snapshot.parser == "official_html"
    assert captured.snapshot.capture_scope == "full_document"
    assert captured.snapshot.content_hash == "sha256:" + hashlib.sha256(PAGE.encode()).hexdigest()
    assert 'result = merge(\n    left, right, on="key"\n)' in captured.element.text
    assert "Obsolete" not in captured.element.text
    assert "navigation" not in captured.element.text
    assert "do_not_include" not in captured.element.text
    assert "Unrelated" not in captured.element.text
    assert "footer" not in captured.element.text
    assert "5\n\n6" in captured.element.text
    assert "left: DataFrame" in captured.element.text
    assert "def helper():\n    value = 1\n\n\n    return value" in captured.element.text
    assert "\n \n" not in captured.element.text
    assert captured.snapshot.source_uri == URL

    revised = PAGE.replace("Merge current tables.", "Merge revised current tables.")
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: response(url, revised))
    refreshed = search("pandas merge usage", requirement=RetrievalRequirement(library="pandas"),
                       previous_hits=result.hits, refresh=True)
    assert all("revised current" in hit.evidence.element.text for hit in refreshed.hits)
    assert captured.snapshot.snapshot_id not in {hit.evidence.snapshot.snapshot_id for hit in refreshed.hits}


@pytest.mark.parametrize("title", ["", "<title>   </title>"])
def test_current_page_without_title_does_not_reuse_cached_version(search, monkeypatch, title):
    page = f"<html><head>{title}</head><body><main>Current merge documentation.</main></body></html>"
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: response(url, page))

    result = search("pandas merge usage", requirement=RetrievalRequirement(library="pandas"))

    assert result.hits
    captured = result.hits[0].evidence
    assert captured.snapshot.parser == "official_html"
    assert captured.snapshot.title == URL
    assert captured.element.text == "Current merge documentation."


@pytest.mark.parametrize("failure", ["timeout", "no_main", "redirect", "cross_library_redirect", "oversized"])
def test_unverified_current_page_keeps_marked_search_excerpt(search, monkeypatch, failure):
    def get(url, **kwargs):
        if failure == "timeout":
            raise requests.Timeout("fixture timeout")
        if failure == "no_main":
            return response(url, "<html><body>Navigation only</body></html>")
        if failure == "redirect":
            return response(url, "", status=302)
        if failure == "cross_library_redirect":
            redirected = response(url, "", status=302)
            redirected.headers["Location"] = "https://numpy.org/doc/stable/reference/index.html"
            return redirected
        return response(url, "<main>" + "x" * (2 * 1024 * 1024) + "</main>")

    monkeypatch.setattr(requests, "get", get)
    result = search("pandas merge usage", requirement=RetrievalRequirement(library="pandas"))
    assert result.hits
    assert result.hits[0].evidence.excerpt == "Obsolete cached description."
    assert "freshness_unverified" in result.hits[0].evidence.snapshot.quality_issues
    assert "official_page_refresh_failed" in result.diagnostics.warnings


def test_explicit_version_does_not_fetch_moving_stable_page(search, monkeypatch):
    def get(url, **kwargs):
        raise AssertionError("Explicit version requests must not use stable-page refresh")
    monkeypatch.setattr(requests, "get", get)
    result = search("pandas 3.0.6 merge", requirement=RetrievalRequirement(library="pandas", version="3.0.6"))
    assert result.hits[0].evidence.snapshot.parser == "tavily"
    assert "freshness_unverified" not in result.hits[0].evidence.snapshot.quality_issues


def test_fragment_variants_share_capture_and_completed_request_reuses_it(search, monkeypatch):
    cached = [{"url": URL + fragment, "title": "Old cached title", "content": "Old cached text.", "score": 0.9}
              for fragment in ("", "#pandas.merge")]
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(
        url, json.dumps({"results": cached}), content_type="application/json"))
    fetched = []

    def get(url, **kwargs):
        fetched.append(url)
        return response(url, PAGE)

    monkeypatch.setattr(requests, "get", get)
    requirement = RetrievalRequirement(library="pandas")
    first = search("pandas merge usage", requirement=requirement)
    second = search("pandas merge usage", requirement=requirement, previous_hits=first.hits,
                    completed_requests=first.diagnostics.completed_requests)

    assert fetched == [URL]
    assert {hit.evidence.snapshot.snapshot_id for hit in first.hits} == {
        hit.evidence.snapshot.snapshot_id for hit in second.hits}
    assert all(hit.evidence.snapshot.parser == "official_html" for hit in second.hits)


def test_stable_page_refresh_has_a_bounded_fetch_count(search, monkeypatch):
    cached = [{"url": URL.replace("pandas.merge.html", f"pandas.operation_{index}.html"),
               "title": "Official reference", "content": "Cached text.", "score": 0.9} for index in range(5)]
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(
        url, json.dumps({"results": cached}), content_type="application/json"))
    fetched = []

    def get(url, **kwargs):
        fetched.append(url)
        return response(url, PAGE)

    monkeypatch.setattr(requests, "get", get)
    result = search("pandas usage", requirement=RetrievalRequirement(library="pandas"))

    assert len(fetched) == 3
    fallback = [hit for hit in result.hits if hit.evidence.snapshot.parser == "tavily"]
    assert fallback
    assert all("freshness_unverified" in hit.evidence.snapshot.quality_issues for hit in fallback)


def test_identical_live_html_from_query_variants_occupies_one_source(search, monkeypatch):
    urls = [URL + "?highlight=pandas+merge", URL + "#pandas.merge"]
    cached = [{"url": url, "title": "Mixed cached versions", "content": "Cached content.", "score": 0.9}
              for url in urls]
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(
        url, json.dumps({"results": cached}), content_type="application/json"))
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: response(url, PAGE))

    result = search("pandas merge usage", requirement=RetrievalRequirement(library="pandas"))

    assert result.hits
    assert len({hit.evidence.snapshot.snapshot_id for hit in result.hits}) == 1
    assert result.hits[0].evidence.snapshot.source_uri in urls


@pytest.mark.parametrize("provider_results", [[], [{
    "url": "https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.merge.html",
    "title": "pandas.merge - pandas 1.5.3 documentation", "content": "Archived merge description.", "score": 0.99,
}]])
def test_requested_stable_api_is_captured_when_search_has_no_current_page(search, monkeypatch, provider_results):
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(
        url, json.dumps({"results": provider_results}), content_type="application/json"))
    fetched = []

    def get(url, **kwargs):
        fetched.append(url)
        return response(url, PAGE)

    monkeypatch.setattr(requests, "get", get)
    result = search("pandas merge usage", requirement=RetrievalRequirement(
        library="pandas", symbols=["pandas.merge"], evidence_needs=["explanation"]), k=1)

    assert result.hits
    captured = result.hits[0].evidence
    assert captured.snapshot.source_uri == URL
    assert captured.snapshot.title == "pandas.merge — pandas 3.1.0 documentation"
    assert captured.snapshot.parser == "official_html"
    assert captured.snapshot.content_hash == "sha256:" + hashlib.sha256(PAGE.encode()).hexdigest()
    assert "Merge current tables." in captured.excerpt
    assert fetched == [URL]


def test_exact_live_api_ranges_precede_adjacent_apis_within_existing_fetch_budget(search, monkeypatch):
    nearby = ["pandas.merge_asof", "pandas.merge_ordered", "pandas.DataFrame.merge", "pandas.DataFrame.combine"]
    cached = [{"url": URL.replace("pandas.merge", name), "title": name,
               "content": "Cached adjacent API description.", "score": 0.99} for name in nearby]
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(
        url, json.dumps({"results": cached}), content_type="application/json"))
    middle = "on: Column names to join on must be found in both DataFrames."
    example = 'result = pandas.merge(left, right, on="key")'
    page = PAGE.replace("<p>Merge current tables.</p>",
        "<p>Merge current tables.</p>" + "<p>Current merge behavior.</p>" * 150
        + f"<p>{middle}</p>" + "<p>Additional merge details.</p>" * 150
        + f"<pre>{example}</pre>")
    fetched = []

    def get(url, **kwargs):
        fetched.append(url)
        return response(url, page if url == URL else PAGE.replace("pandas.merge", "pandas.merge_asof"))

    monkeypatch.setattr(requests, "get", get)
    result = search("pandas merge usage", requirement=RetrievalRequirement(
        library="pandas", symbols=["pandas.merge"], evidence_needs=["explanation"]), k=1, refresh=True)
    exact = [hit for hit in result.hits if hit.evidence.snapshot.source_uri == URL]

    assert exact
    assert middle in "\n".join(hit.evidence.excerpt for hit in exact)
    assert example in "\n".join(hit.evidence.excerpt for hit in exact)
    assert [hit.rank for hit in exact] == list(range(1, len(exact) + 1))
    assert fetched[0] == URL
    assert len(fetched) == 3
    assert len(result.hits) <= 6


def test_unavailable_direct_api_does_not_relabel_archived_evidence(search, monkeypatch):
    archived_url = "https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.merge.html"
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": [{
        "url": archived_url, "title": "pandas.merge - pandas 1.5.3 documentation",
        "content": "Archived merge description.", "score": 0.9,
    }]}), content_type="application/json"))
    fetched = []

    def get(url, **kwargs):
        fetched.append(url)
        return response(url, "Not found", status=404)

    monkeypatch.setattr(requests, "get", get)
    requirement = RetrievalRequirement(library="pandas", symbols=["pandas.merge"])
    fallback = search("pandas merge usage", requirement=requirement, k=1)

    assert fetched == [URL]
    assert fallback.hits
    assert all(hit.evidence.snapshot.source_uri == archived_url for hit in fallback.hits)
    assert all(hit.evidence.snapshot.parser == "tavily" for hit in fallback.hits)
    assert "official_page_refresh_failed" in fallback.diagnostics.warnings
    fetched.clear()
    pinned = search("pandas 1.5.3 merge usage", requirement=requirement.model_copy(update={"version": "1.5.3"}), k=1)
    assert fetched == []
    assert all(hit.evidence.snapshot.source_uri == archived_url for hit in pinned.hits)
