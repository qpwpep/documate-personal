"""Stable documentation is captured from its current HTTP page, not search cache."""

import hashlib
import json
from urllib.parse import urldefrag

import pytest
import requests

from src.core.planner_schema import PlannerOutput, RetrievalRequirement, RetrievalTask
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.sections import parse_doc_sections
from src.infra.tools.docs_search.tool import build_docs_search_tool
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
from src.runtime.nodes.synthesis.evidence_selection import prepare_evidence_candidates


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
    assert result.hits[0].evidence.snapshot.source_uri in {urldefrag(url)[0] for url in urls}


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


@pytest.mark.parametrize("library,symbol,version,target_url,archived_url", [
    ("pandas", "pandas.merge", "1.5.3", URL,
     "https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.merge.html"),
    ("numpy", "numpy.reshape", "2.0", "https://numpy.org/doc/stable/reference/generated/numpy.reshape.html",
     "https://numpy.org/doc/2.0/reference/generated/numpy.reshape.html"),
])
def test_unavailable_direct_api_does_not_relabel_archived_evidence(
    search, monkeypatch, library, symbol, version, target_url, archived_url,
):
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": [{
        "url": archived_url, "title": f"{symbol} - {library} {version} documentation",
        "content": "Archived API description.", "score": 0.9,
    }]}), content_type="application/json"))
    fetched = []

    def get(url, **kwargs):
        fetched.append(url)
        return response(url, "Not found", status=404)

    monkeypatch.setattr(requests, "get", get)
    requirement = RetrievalRequirement(library=library, symbols=[symbol])
    fallback = search(f"{symbol} usage", requirement=requirement, k=1)

    assert fetched == [target_url]
    assert fallback.hits
    assert all(hit.evidence.snapshot.source_uri == archived_url for hit in fallback.hits)
    assert all(hit.evidence.snapshot.parser == "tavily" for hit in fallback.hits)
    assert "official_page_refresh_failed" in fallback.diagnostics.warnings
    fetched.clear()
    pinned = search(f"{symbol} {version} usage", requirement=requirement.model_copy(update={"version": version}), k=1)
    assert fetched == []
    assert all(hit.evidence.snapshot.source_uri == archived_url for hit in pinned.hits)


@pytest.mark.parametrize(("locale", "module", "member", "description"), [
    ("ko/", "pathlib", "Path.exists", "Return True if the path points to an existing file or directory and False otherwise."),
    ("", "functools", "lru_cache", "Return a wrapper that caches calls to the decorated function."),
])
def test_discovered_stdlib_module_seeds_current_page_before_unrelated_stable_windows(
    search, monkeypatch, locale, module, member, description,
):
    current_url = f"https://docs.python.org/{locale}3/library/{module}.html"
    archived = {"url": f"https://docs.python.org/{locale}3.9/library/{module}.html",
                "title": f"{module} — Python 3.9 documentation",
                "raw_content": f"{member}()\nArchived behavior.", "score": 0.9}
    unrelated = [{"url": f"https://docs.python.org/{locale}3/whatsnew/3.{version}.html",
                  "title": f"What's new in Python 3.{version}",
                  "raw_content": "Other Python release details.\n" * 500, "score": 0.8}
                 for version in (7, 8)]
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(
        url, json.dumps({"results": [archived, *unrelated]}), content_type="application/json"))
    current_page = (f"<html><head><title>{module} — Python 3.15 documentation</title></head><body><main>"
                    + "<p>Other module details.</p>" * 200
                    + f'<dl><dt id="{module}.{member}">{member}()'
                    + f'<a href="#{module}.{member}" title="Link to this definition">¶</a></dt>'
                    + f"<dd><p>{description}</p></dd></dl></main></body></html>")
    fetched = []

    def get(url, **kwargs):
        fetched.append(url)
        page = current_page if url == current_url else "<main>" + "<p>Unrelated release details.</p>" * 300 + "</main>"
        return response(url, page)

    monkeypatch.setattr(requests, "get", get)
    task = RetrievalTask(route="docs", query=f"Python {module}.{member} behavior", k=3,
                         requirement={"library": "Python", "symbols": [f"{module}.{member}"],
                                      "evidence_needs": ["explanation"]})
    result = search(task.query, requirement=task.requirement, k=task.k)
    fresh = [hit for hit in result.hits if hit.evidence.snapshot.source_uri == current_url]

    assert fresh
    assert all(hit.evidence.snapshot.parser == "official_html" for hit in fresh)
    assert all(hit.evidence.snapshot.content_hash == "sha256:" + hashlib.sha256(current_page.encode()).hexdigest()
               for hit in fresh)
    assert all("Archived behavior." not in hit.evidence.element.text for hit in fresh)
    candidates = prepare_evidence_candidates(user_input=task.query,
        hits=[hit.model_copy(update={"requirement_id": task.requirement_id}) for hit in result.hits],
        planner_output=PlannerOutput(use_retrieval=True, tasks=[task]),
        budget=RetrievedEvidenceBudget(1800, 6000, 8))
    assert any(hit.evidence.snapshot.source_uri == current_url and description in hit.evidence.excerpt
               for hit in candidates)
    assert fetched[1] == current_url
    assert len([url for url in fetched if not url.endswith("objects.inv")]) == 3


@pytest.mark.parametrize("failure", ["timeout", "redirect", "oversized"])
def test_failed_stdlib_current_capture_preserves_archived_source(search, monkeypatch, failure):
    archived_url = "https://docs.python.org/ko/3.9/library/pathlib.html"
    original = "Path.exists()\nArchived existence behavior."
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": [{
        "url": archived_url, "title": "pathlib — Python 3.9 documentation", "raw_content": original, "score": 0.9,
    }]}), content_type="application/json"))
    fetched = []

    def get(url, **kwargs):
        fetched.append(url)
        if failure == "timeout":
            raise requests.Timeout("fixture timeout")
        if failure == "redirect":
            redirected = response(url, "", status=302)
            redirected.headers["Location"] = "https://docs.python.org.attacker.example/3/library/pathlib.html"
            return redirected
        return response(url, "<main>" + "x" * (2 * 1024 * 1024) + "</main>")

    monkeypatch.setattr(requests, "get", get)
    requirement = RetrievalRequirement(library="Python", symbols=["pathlib.Path.exists"])
    result = search("Python pathlib.Path.exists", requirement=requirement)

    assert fetched == ["https://docs.python.org/ko/3/objects.inv", "https://docs.python.org/ko/3/library/pathlib.html"]
    assert result.hits
    assert all(hit.evidence.snapshot.source_uri == archived_url for hit in result.hits)
    assert all(hit.evidence.snapshot.parser == "tavily" and hit.evidence.element.text == original for hit in result.hits)
    assert "official_page_refresh_failed" in result.diagnostics.warnings
    fetched.clear()
    pinned = search("Python 3.9 pathlib.Path.exists", requirement=requirement.model_copy(update={"version": "3.9"}))
    assert fetched == ["https://docs.python.org/ko/3.9/objects.inv", archived_url]
    assert all(hit.evidence.snapshot.source_uri == archived_url for hit in pinned.hits)


@pytest.mark.parametrize("symbol,description", [
    ("numpy.reshape", "One shape dimension can be -1; its length is inferred from the array size and remaining dimensions."),
    ("numpy.mean", "Compute the arithmetic mean along the specified axis."),
])
def test_numpy_target_page_is_captured_when_search_returns_only_adjacent_archives(
    search, monkeypatch, symbol, description,
):
    target_url = f"https://numpy.org/doc/stable/reference/generated/{symbol}.html"
    archived_url = "https://numpy.org/doc/2.0/reference/generated/numpy.ndarray.size.html"
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": [{
        "url": archived_url, "title": "numpy.ndarray.size — NumPy v2.0 Manual",
        "raw_content": "Number of elements in the array.", "score": 0.99,
    }]}), content_type="application/json"))
    page = f"<title>{symbol} — NumPy stable documentation</title><main><h1>{symbol}</h1><p>{description}</p></main>"
    fetched = []

    def get(url, **kwargs):
        fetched.append(url)
        return response(url, page)

    monkeypatch.setattr(requests, "get", get)
    result = search(f"{symbol} behavior", requirement=RetrievalRequirement(
        library="numpy", symbols=[symbol], evidence_needs=["explanation"]), k=1)

    assert fetched == [target_url]
    assert result.hits[0].evidence.snapshot.source_uri == target_url
    assert result.hits[0].evidence.snapshot.parser == "official_html"
    assert description in result.hits[0].evidence.excerpt


def test_explicit_python_version_captures_discovered_module_at_requested_version(search, monkeypatch):
    discovered_url = "https://docs.python.org/ko/3/library/pathlib.html"
    target_url = "https://docs.python.org/ko/3.9/library/pathlib.html"
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": [{
        "url": discovered_url, "title": "pathlib — Python 3.15 documentation",
        "raw_content": "Path.exists(*, follow_symlinks=True)\nCurrent behavior.", "score": 0.99,
    }]}), content_type="application/json"))
    body = "Path.exists()\nReturn True if the path points to an existing file or directory. Other errors are propagated."
    page = f"<title>pathlib — Python 3.9 documentation</title><main><pre>{body}</pre></main>"
    fetched = []

    def get(url, **kwargs):
        fetched.append(url)
        return response(url, page)

    monkeypatch.setattr(requests, "get", get)
    result = search("Python 3.9 pathlib.Path.exists", requirement=RetrievalRequirement(
        library="Python", symbols=["pathlib.Path.exists"], version="3.9", evidence_needs=["explanation"]), k=1)

    assert fetched == ["https://docs.python.org/ko/3.9/objects.inv", target_url]
    assert result.hits[0].evidence.snapshot.source_uri == target_url
    assert result.hits[0].evidence.snapshot.title == "pathlib — Python 3.9 documentation"
    assert result.hits[0].evidence.snapshot.parser == "official_html"
    assert result.hits[0].evidence.excerpt == f"```\n{body}\n```"
    assert "follow_symlinks" not in result.hits[0].evidence.excerpt


def test_html_example_definition_does_not_take_ownership_of_following_api_prose(search, monkeypatch):
    url = "https://docs.python.org/3/library/functools.html"
    condition = "If maxsize is set to None, the LRU feature is disabled and the cache can grow without bound."
    code = ("@lru_cache\ndef count_vowels(word):\n"
            + "    # Retain this example's original whitespace and blank lines.\n" * 35
            + "\n\n    return sum(word.count(vowel) for vowel in 'AEIOUaeiou')\n")
    page = ("<title>functools — Python documentation</title><main>"
            "<dl><dt>@functools.lru_cache(maxsize=128, typed=False)¶</dt><dd>"
            "<p>Decorator that caches calls to the wrapped function.</p>"
            f"<pre><code>{code}</code></pre><p>{condition}</p></dd></dl>"
            "<dl><dt>@functools.total_ordering¶</dt><dd><p>Supply ordering methods.</p></dd></dl></main>")
    monkeypatch.setattr(requests, "post", lambda request_url, **kwargs: response(request_url, json.dumps({"results": [{
        "url": url, "title": "functools — Python documentation", "content": "Cached excerpt.", "score": 0.9,
    }]}), content_type="application/json"))
    monkeypatch.setattr(requests, "get", lambda request_url, **kwargs: response(request_url, page))
    task = RetrievalTask(route="docs", query="functools.lru_cache maxsize=None", k=3, requirement={
        "library": "Python", "symbols": ["functools.lru_cache"], "aspects": ["maxsize", "None"],
        "evidence_needs": ["explanation"],
    })

    result = search(task.query, requirement=task.requirement)
    evidence = result.hits[0].evidence
    text = evidence.element.text
    sections = parse_doc_sections(text, title=evidence.snapshot.title, url=evidence.snapshot.source_uri)
    owned = next(section for section in sections if condition in text[section.start:section.end])
    assert owned.owner == "functools.lru_cache"
    assert code in text
    assert any(section.owner == "functools.total_ordering" for section in sections)
    prepared = prepare_evidence_candidates(user_input=task.query,
        hits=[hit.model_copy(update={"requirement_id": task.requirement_id}) for hit in result.hits],
        planner_output=PlannerOutput(use_retrieval=True, tasks=[task]),
        budget=RetrievedEvidenceBudget(1800, 6000, 8))
    assert any(condition in hit.evidence.excerpt for hit in prepared)


def test_discovered_archived_guide_is_captured_at_stable_without_relabeling_it(search, monkeypatch):
    archived = "https://pandas.pydata.org/pandas-docs/version/2.1/user_guide/merging.html"
    stable = "https://pandas.pydata.org/docs/user_guide/merging.html"
    archived_text = "Archived pandas.merge guide. Duplicate keys raise MergeError."
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: response(url, json.dumps({"results": [
        {"url": URL + "?highlight=merge", "title": "merge", "content": "Cached API description.", "score": 0.9},
        {"url": archived, "title": "Merging — pandas 2.1", "content": archived_text, "score": 0.8},
    ]}), content_type="application/json"))
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        text = ("<title>Merging — pandas 3.1</title><main><h1>Merging</h1>"
                "<p>pandas.merge(validate='one_to_one') checks uniqueness in both datasets.</p>"
                "<pre>MergeError: Merge keys are not unique in left dataset; not a one-to-one merge</pre></main>")
        return response(url, text if url == stable else PAGE)

    monkeypatch.setattr(requests, "get", get)
    requirement = RetrievalRequirement(library="pandas", symbols=["pandas.merge"], aspects=["validate", "one_to_one"])

    result = search("pandas.merge duplicate keys validation", requirement=requirement, k=5)

    fresh = [hit for hit in result.hits if hit.evidence.snapshot.source_uri == stable]
    assert fresh
    assert all(hit.evidence.snapshot.title == "Merging — pandas 3.1" for hit in fresh)
    assert any("MergeError: Merge keys" in hit.evidence.excerpt for hit in fresh)
    assert calls[:2] == [URL, stable]
    assert len(calls) <= 3
    calls.clear()

    pinned = search("pandas 2.1 merge validation", requirement=requirement.model_copy(update={"version": "2.1"}))

    assert calls == []
    assert any(hit.evidence.snapshot.source_uri == archived and hit.evidence.excerpt == archived_text
               for hit in pinned.hits)
