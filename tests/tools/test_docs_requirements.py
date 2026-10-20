"""Docs retrieval preserves official source candidates for later semantic decisions."""
import pytest
import requests

from src.core.planner_schema import RetrievalRequirement
from src.core.evidence import SearchHit
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool


def result(symbol="numpy.reshape", *, version="2.4", body=None):
    return {
        "url": f"https://numpy.org/doc/{version}/reference/generated/{symbol}.html",
        "title": symbol,
        "raw_content": body or f"# {symbol}\n{symbol}(a, shape, order='C')\nParameters\norder: Read elements in C or F index order.",
        "content": f"{symbol} reference.",
        "score": 0.9,
    }


@pytest.fixture
def search(monkeypatch):
    requests_seen = []
    batches = []

    def post(url, *, json, **kwargs):
        requests_seen.append(json)
        response = requests.Response()
        response.status_code = 200
        response._content = __import__("json").dumps({"results": batches.pop(0) if batches else []}).encode()
        return response

    monkeypatch.setattr(requests, "post", post)
    tool = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test"))
    return tool, batches, requests_seen


@pytest.mark.parametrize("previous_scope", ["other_library", "other_version"])
def test_previous_candidates_keep_library_scope_and_defer_version_relevance(search, previous_scope):
    """Library domains constrain sources; requested versions are judged by Decisions."""
    tool, batches, seen = search
    old = result("numpy.concatenate", version="2.4")
    old_requirement = RetrievalRequirement(library='numpy')
    if previous_scope == "other_library":
        old["url"] = "https://pandas.pydata.org/docs/user_guide/merging.html"
        old_requirement = RetrievalRequirement(library='pandas')
    batches.append([old])
    first = tool("official reference", requirement=old_requirement)
    previous = [SearchHit.model_validate(hit) for hit in first.hits]
    assert previous
    batches.append([result("numpy.stack", version="1.26")])

    requirement = RetrievalRequirement(library='numpy', symbols=['numpy.concatenate', 'numpy.stack'], version='1.26' if previous_scope == 'other_version' else None)
    second = tool("numpy.concatenate numpy.stack 1.26", requirement=requirement, previous_hits=previous)

    assert second.diagnostics.answerability == "unknown"
    assert second.diagnostics.missing_requirements == []
    assert {hit.evidence.snapshot.title for hit in second.hits} == (
        {"numpy.stack"} if previous_scope == "other_library" else {"numpy.concatenate", "numpy.stack"}
    )
    assert len(seen) == 2


@pytest.mark.parametrize("refresh_result", ["same_snapshot", "wrong_owner", "timeout"])
def test_explicit_refresh_preserves_prior_candidates_and_does_not_reject_other_symbols(search, monkeypatch, refresh_result):
    tool, batches, seen = search
    query = "numpy.reshape explanation"
    source = result(body="# numpy.reshape\nnumpy.reshape(a, shape)\nGives a new shape to an array.")
    requirement = RetrievalRequirement(library="numpy", symbols=["numpy.reshape"], evidence_needs=["explanation"])
    batches.append([source])
    first = tool(query, requirement=requirement)
    assert first.diagnostics.answerability == "unknown"

    if refresh_result == "timeout":
        def timeout(url, *, json, **kwargs):
            seen.append(json)
            raise requests.Timeout("temporary search failure")
        monkeypatch.setattr(requests, "post", timeout)
    else:
        batches.append([source] if refresh_result == "same_snapshot" else [result("numpy.stack")])

    refreshed = tool(query, requirement=requirement, previous_hits=first.hits,
                     completed_requests=first.diagnostics.completed_requests, refresh=True)

    assert [request["query"] for request in seen] == [query, query]
    first_ids = {hit.evidence.id for hit in first.hits}
    refreshed_ids = {hit.evidence.id for hit in refreshed.hits}
    assert first_ids <= refreshed_ids
    if refresh_result == "wrong_owner":
        assert {hit.evidence.snapshot.title for hit in refreshed.hits} == {"numpy.reshape", "numpy.stack"}
    else:
        assert refreshed_ids == first_ids
    assert refreshed.diagnostics.answerability == "unknown"
    assert refreshed.diagnostics.missing_requirements == []
    assert len(refreshed.diagnostics.provider_requests) == 1
    assert refreshed.diagnostics.provider_requests[0].status == ("error" if refresh_result == "timeout" else "completed")
    assert refreshed.diagnostics.error_code == ("RETRIEVAL_DOCS_TIMEOUT" if refresh_result == "timeout" else None)


@pytest.mark.parametrize(("second_k", "expected_calls"), [(2, 1), (3, 2)])
def test_completed_query_reuses_candidates_only_with_sufficient_provider_limit(search, second_k, expected_calls):
    tool, batches, seen = search
    query = "numpy.reshape order"
    requirement = RetrievalRequirement(library="numpy", symbols=["numpy.reshape"], aspects=["order"])
    batches.append([result()])
    first = tool(query, k=1, requirement=requirement)
    batches.append([result()])

    second = tool(query, k=second_k, requirement=requirement, previous_hits=first.hits,
                  completed_requests=first.diagnostics.completed_requests)

    assert len(seen) == expected_calls
    assert seen[-1]["max_results"] == min(20, max(6, second_k * 3))
    assert second.hits
    assert second.diagnostics.answerability == "unknown"
    assert len(second.diagnostics.provider_requests) == expected_calls - 1


@pytest.mark.parametrize(("library", "url", "counter"), [
    ("numpy", "https://numpy.org.attacker.example/doc/stable/reference/generated/numpy.reshape.html", "filtered_invalid_url_count"),
    ("numpy", "https://numpy.org/community/numpy.reshape.html", "filtered_path_prefix_count"),
    ("numpy", "https://pandas.pydata.org/docs/reference/api/numpy.reshape.html", "filtered_cross_domain_count"),
])
def test_provider_sources_keep_official_document_and_library_scope(search, library, url, counter):
    tool, batches, _seen = search
    source = result()
    source["url"] = url
    batches.append([source])
    payload = tool("numpy.reshape", requirement=RetrievalRequirement(library=library, symbols=["numpy.reshape"]))
    assert payload.hits == []
    assert payload.diagnostics.answerability == "missing"
    assert payload.diagnostics.missing_requirements == []
    assert getattr(payload.diagnostics, counter) == 1


def python_reference(body):
    return {
        "url": "https://docs.python.org/3/library/pathlib.html",
        "title": "pathlib — Object-oriented filesystem paths",
        "content": "Path objects represent filesystem paths.",
        "raw_content": body,
        "score": 0.9,
    }


def test_korean_prose_without_query_terms_remains_a_candidate(search):
    tool, batches, _seen = search
    description = "최대 크기를 지정하지 않은 캐시는 새 항목을 계속 저장합니다."
    source = python_reference(description)
    source["url"] = "https://docs.python.org/ko/3/library/functools.html"
    source["title"] = "functools — 고차 함수와 호출 가능한 객체에 대한 연산"
    batches.append([source])

    payload = tool("functools.lru_cache maxsize=None 동작 설명", requirement=RetrievalRequirement(
        library="Python", symbols=["functools.lru_cache"], aspects=["maxsize", "None"],
        evidence_needs=["parameters", "explanation"], match="definition",
    ))

    assert payload.diagnostics.answerability == "unknown"
    assert payload.diagnostics.missing_requirements == []
    assert len(payload.hits) == 1
    evidence = payload.hits[0].evidence
    assert evidence.element.text == description
    assert evidence.excerpt == description


def test_large_source_candidates_preserve_return_conditions_and_release_metadata(search):
    """Candidate construction keeps late source text without deciding its sufficiency."""
    tool, batches, _seen = search
    symbol = "exists"
    description = "Return `True` if the path points to an existing file or directory and `False` if the path is invalid, inaccessible or missing."
    # The declaration, fenced example and release annotation have the structure
    # returned by Tavily for the live Python documentation.
    body = (
        "# Overview\n" + "A path object represents a location in a filesystem.\n" * 75
        + f"\nPath.{symbol}(*\\**, *follow\\_symlinks=True*)[¶](#pathlib.Path.{symbol} \"Link to this definition\")\n"
        + f":   {description}\n\n"
        + "    This method normally follows symlinks; to check the symlink itself, add the argument `follow_symlinks=False`.\n\n"
        + f"    ```\n    >>> Path('example.txt').{symbol}()\n    True\n    ```\n"
        + "\n    Changed in version 3.12: The *follow\\_symlinks* parameter was added.\n\n"
        + f"Additional details\nFor symbolic links, Path.{symbol}() checks the target.\n\n"
        + "# Path.resolve\nPath.resolve()\nReturn an absolute path with all symlinks resolved.\n"
    )
    batches.append([python_reference(body)])

    payload = tool(f"Python pathlib.Path.{symbol} 파일 디렉터리 존재 여부 설명", k=1,
                   requirement=RetrievalRequirement(library="Python", symbols=[f"pathlib.Path.{symbol}"],
                                                    evidence_needs=["explanation"]))

    assert payload.diagnostics.answerability == "unknown"
    assert payload.diagnostics.missing_requirements == []
    assert 1 <= len(payload.hits) <= 6
    assert any(description in hit.evidence.excerpt for hit in payload.hits)
    assert any("follow_symlinks=False" in hit.evidence.excerpt for hit in payload.hits)
    assert any("Changed in version 3.12" in hit.evidence.excerpt for hit in payload.hits)
    for hit in payload.hits:
        evidence = hit.evidence
        assert evidence.element.text == body
        assert evidence.excerpt == body[evidence.selection.start:evidence.selection.end]
