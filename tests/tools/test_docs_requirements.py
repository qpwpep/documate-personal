"""Evidence requirements are checked against source identity and returned text."""
import pytest
import requests

from src.core.evidence import SearchHit
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool


# Public NumPy response captured during the paid API verification; contains no user data.
RECORDED_NUMPY_RESHAPE = {'url': 'https://numpy.org/doc/2.4/reference/generated/numpy.reshape.html', 'title': 'numpy.reshape — NumPy v2.4 Manual', 'raw_content': '[![NumPy v2.4 Manual - Home](../../_static/numpylogo.svg) ![NumPy v2.4 Manual - Home](../../_static/numpylogo_dark.svg)](../../index.html)\n\n* [GitHub](https://github.com/numpy/numpy "GitHub")\n\n# numpy.reshape[#](#numpy-reshape "Link to this heading")\n\nnumpy.reshape(*a*, */*, *shape*, *order=\'C\'*, *\\**, *copy=None*)[[source]](https://github.com/numpy/numpy/blob/v2.4.0/numpy/_core/fromnumeric.py#L207-L299)[#](#numpy.reshape "Link to this definition")\n:   Gives a new shape to an array without changing its data.\n\n    Parameters:\n    :   **a**array\\_like\n        :   Array to be reshaped.\n\n        **shape**int or tuple of ints\n        :   The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.\n\n        **order**{‘C’, ‘F’, ‘A’}, optional\n        :   Read the elements of `a` using this index order, and place the elements into the reshaped array using this index order. ‘C’ means to read / write the elements using C-like index order, with the last axis index changing fastest, back to the first axis index changing slowest. ‘F’ means to read / write the elements using Fortran-like index order, with the first index changing fastest, and the last index changing slowest. Note that the ‘C’ and ‘F’ options take no account of the memory layout of the underlying array, and only refer to the order of indexing. ‘A’ means to read / write the elements in Fortran-like index order if `a` is Fortran *contiguous* in memory, C-like order otherwise.\n\n        **copy**bool, optional\n        :   If `True`, then the array data is copied. If `None`, a copy will only be made if it’s required by `order`. For `False` it raises a `ValueError` if a copy cannot be avoided. Default: `None`.\n\n    Returns:\n    :   **reshaped\\_array**ndarray\n        :   This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the *memory layout* (C- or Fortran- contiguous) of the returned array.\n\n    See also\n\n    [`ndarray.reshape`](numpy.ndarray.reshape.html#numpy.ndarray.reshape "numpy.ndarray.reshape")\n    :   Equivalent method.\n\n    Notes\n\n    It is not always possible to change the shape of an array without copying the data.\n\n    The `order` keyword gives the index ordering both for *fetching* the values from `a`, and then *placing* the values into the output array. For example, let’s say you have an array:\n\n    ```\n    >>> a = np. arange(6). reshape((3, 2))>>> aarray([[0, 1], [2, 3], [4, 5]])\n    ```\n\n    You can think of reshaping as first raveling the array (using the given index order), then inserting the elements from the raveled array into the new array using the same kind of index ordering as was used for the raveling.\n\n    ```\n    >>> np. reshape(a,(2, 3))# C-like index orderingarray([[0, 1, 2], [3, 4, 5]])>>> np. reshape(np. ravel(a),(2, 3)) # equivalent to C ravel then C reshapearray([[0, 1, 2], [3, 4, 5]])>>> np. reshape(a,(2, 3), order = \'F\')# Fortran-like index orderingarray([[0, 4, 3], [2, 1, 5]])>>> np. reshape(np. ravel(a, order = \'F\'),(2, 3), order = \'F\')array([[0, 4, 3], [2, 1, 5]])\n    ```\n\n    Examples\n\n    ```\n    >>> import  numpy  as  np>>> a = np. array([[1, 2, 3],[4, 5, 6]])>>> np. reshape(a, 6)array([1, 2, 3, 4, 5, 6])>>> np. reshape(a, 6, order = \'F\')array([1, 4, 2, 5, 3, 6])\n    ```\n\n    ```\n    >>> np. reshape(a,(3, - 1)) # the unspecified value is inferred to be 2array([[1, 2], [3, 4], [5, 6]])\n    ```\n\nOn this page\n\n ', 'content': 'numpy.reshape reference.', 'score': 0.9}

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

    def head(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "head", head)
    from src.infra.tools.docs_search.url_validation import validate_doc_url
    validate_doc_url.cache_clear()
    tool = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test"))
    return tool, batches, requests_seen


def test_numpy_alias_uses_canonical_definition_without_losing_source_text(search):
    """The np alias resolves to the NumPy definition in the original provider source."""
    tool, batches, seen = search
    source = RECORDED_NUMPY_RESHAPE
    batches.append([source])
    payload = tool("NumPy np.reshape order", requirement={"library": "NumPy", "symbols": ["np.reshape"], "aspects": ["order"], "match": "definition"})
    hit = SearchHit.model_validate(payload["hits"][0])
    assert payload["diagnostics"]["answerability"] == "covered"
    assert hit.evidence.element.text == source["raw_content"]
    assert hit.evidence.excerpt in source["raw_content"]
    assert seen[0]["include_domains"] == ["numpy.org"]
    assert len(seen) == 1


def test_explicit_library_wins_over_ambiguous_pipeline_hint(search):
    """A Hugging Face requirement cannot be searched in scikit-learn."""
    tool, batches, seen = search
    batches.append([{"url": "https://huggingface.co/docs/transformers/main_classes/pipelines", "title": "Pipelines", "content": "Pipelines run model inference."}])
    payload = tool("Hugging Face pipeline documentation", requirement={"library": "Hugging Face"})
    assert payload["diagnostics"]["answerability"] == "covered"
    assert [r["include_domains"] for r in seen] == [["huggingface.co"]]


def test_incidental_symbol_mention_does_not_satisfy_definition(search):
    """A document defining another function is not the requested API definition."""
    tool, batches, seen = search
    batches.append([result("numpy.array", body="# numpy.array\nnumpy.array(object)\nSee also numpy.reshape for changing shape.")])
    payload = tool("numpy.reshape definition", requirement={"library": "numpy", "symbols": ["numpy.reshape"], "match": "definition"})
    assert payload["diagnostics"]["answerability"] == "missing"
    assert payload["hits"] == []
    assert len(seen) == 2


def test_version_identity_is_not_replaced_by_a_valid_stable_alias(search):
    """A requested historical version rejects current-version source text."""
    tool, batches, seen = search
    batches.append([result(version="2.4")])
    payload = tool("numpy.reshape 1.26 order", requirement={"library": "numpy", "symbols": ["numpy.reshape"], "version": "1.26", "match": "definition"})
    assert payload["diagnostics"]["answerability"] == "missing"
    assert "version:1.26" in payload["diagnostics"]["missing_requirements"]
    assert payload["hits"] == []


def test_missing_aspect_preserves_partial_evidence(search):
    """An absent parameter is recorded as missing while usable definition evidence survives."""
    tool, batches, seen = search
    batches.append([result()])
    payload = tool("numpy.reshape copy", requirement={"library": "numpy", "symbols": ["numpy.reshape"], "aspects": ["copy"], "match": "definition"})
    assert payload["diagnostics"]["answerability"] == "partial"
    assert payload["diagnostics"]["missing_requirements"] == ["aspect:copy"]
    assert len(payload["hits"]) == 1


def test_empty_search_reformulates_only_the_requested_subject_and_skips_attempts(search):
    """Retries preserve the subject and do not purchase a search already attempted."""
    tool, batches, seen = search
    first = tool("numpy.reshape order", requirement={"library": "numpy", "symbols": ["numpy.reshape"], "aspects": ["order"]})
    assert len(seen) == 2
    assert all("numpy.reshape" in r["query"] and "order" in r["query"] for r in seen)
    second = tool("numpy.reshape order", requirement={"library": "numpy", "symbols": ["numpy.reshape"], "aspects": ["order"]}, attempted_queries=first["diagnostics"]["attempted_queries"])
    assert len(seen) == 2
    assert second["diagnostics"]["answerability"] == "missing"


def test_requested_aspect_near_end_of_source_is_selected_with_exact_ranges(search):
    """A parameter beyond the first window is selected from the actual source."""
    tool, batches, seen = search
    source = result(body="# numpy.reshape\nnumpy.reshape(a, shape)\n" + ("Background details.\n" * 300) + "copy: Return an independent copy when True.\n")
    batches.append([source])
    payload = tool("numpy.reshape copy", requirement={"library": "numpy", "symbols": ["numpy.reshape"], "aspects": ["copy"], "match": "definition"})
    assert payload["diagnostics"]["answerability"] == "covered"
    hits = [SearchHit.model_validate(h) for h in payload["hits"]]
    assert any("copy: Return" in h.evidence.excerpt for h in hits)
    assert all(h.evidence.excerpt in source["raw_content"] for h in hits)


def test_unknown_explicit_library_does_not_search_unrelated_allowlisted_sites(search):
    """An unsupported official source is reported instead of substituting other libraries."""
    tool, batches, seen = search
    payload = tool("ExampleNewLibrary normalize", requirement={"library": "ExampleNewLibrary"})
    assert payload["diagnostics"]["answerability"] == "missing"
    assert payload["diagnostics"]["missing_requirements"] == ["library:ExampleNewLibrary"]
    assert payload["hits"] == []
    assert seen == []


def test_topic_request_rejects_a_library_page_without_the_requested_topic(search):
    """A library match alone cannot answer an unrelated topic question."""
    tool, batches, seen = search
    batches.append([result("numpy.array", body="# numpy.array\nCreate an array from an object.")])
    payload = tool("numpy broadcasting", requirement={"library": "numpy", "aspects": ["broadcasting"]})
    assert payload["diagnostics"]["answerability"] == "missing"
    assert payload["hits"] == []


def test_each_required_symbol_has_a_definition_when_result_budget_allows(search):
    """Result selection covers separate required API definitions before repeated versions."""
    tool, batches, seen = search
    batches.append([result(version="2.4"), result(version="2.0"), result("numpy.concatenate")])
    payload = tool("numpy.reshape numpy.concatenate", k=2,
                   requirement={"library": "numpy", "symbols": ["numpy.reshape", "numpy.concatenate"]})
    assert payload["diagnostics"]["answerability"] == "covered"
    assert {h["evidence"]["snapshot"]["title"] for h in payload["hits"]} == {"numpy.reshape", "numpy.concatenate"}


def test_unrelated_page_call_example_is_not_an_api_definition(search):
    """A call in another API's example does not establish the requested definition."""
    tool, batches, seen = search
    batches.append([result("numpy.array", body="# numpy.array\nCreate an array.\nExamples\nnumpy.reshape(a, shape)\n")])
    payload = tool("numpy.reshape", requirement={"library": "numpy", "symbols": ["numpy.reshape"], "match": "definition"})
    assert payload["diagnostics"]["answerability"] == "missing"
    assert payload["hits"] == []


@pytest.mark.parametrize(("requested", "returned"), [
    ("numpy.ndarray", "numpy.ndarray.sort"),
    ("numpy.RESHAPE", "numpy.reshape"),
])
def test_api_ownership_preserves_symbol_boundary_and_case(search, requested, returned):
    """A different API or differently cased name cannot fulfill the exact symbol requirement."""
    tool, batches, seen = search
    batches.append([result(returned)])
    payload = tool(requested, requirement={"library": "numpy", "symbols": [requested], "match": "definition"})
    assert payload["diagnostics"]["answerability"] == "missing"
    assert payload["diagnostics"]["missing_requirements"] == [f"symbol:{requested}"]
    assert payload["hits"] == []


def test_new_search_combines_previous_partial_evidence_for_the_same_requirement(search):
    """A later search can fulfill a missing symbol without discarding the first definition."""
    tool, batches, seen = search
    requirement = {"library": "numpy", "symbols": ["numpy.concatenate", "numpy.stack"]}
    batches.append([result("numpy.concatenate")])
    first = tool("numpy concatenate and stack", requirement=requirement)
    assert first["diagnostics"]["answerability"] == "partial"
    previous = [SearchHit.model_validate(hit) for hit in first["hits"]]
    original = [hit.model_dump(mode="json") for hit in previous]
    batches.append([result("numpy.stack")])

    second = tool("numpy.stack reference", requirement=requirement, previous_hits=previous,
                  attempted_queries=first["diagnostics"]["attempted_queries"])

    assert second["diagnostics"]["answerability"] == "covered"
    assert second["diagnostics"]["missing_requirements"] == []
    assert {hit["evidence"]["snapshot"]["title"] for hit in second["hits"]} == {"numpy.concatenate", "numpy.stack"}
    assert previous[0].evidence.model_dump(mode="json") in [hit["evidence"] for hit in second["hits"]]
    assert [hit.model_dump(mode="json") for hit in previous] == original
    assert second["diagnostics"]["provider_result_count"] == 1
    assert len(seen) == 3


@pytest.mark.parametrize("previous_scope", ["other_library", "other_version"])
def test_previous_partial_evidence_is_rechecked_against_library_and_version(search, previous_scope):
    """Preserved evidence from a different source scope cannot satisfy a new requirement."""
    tool, batches, seen = search
    old = result("numpy.concatenate", version="2.4")
    old_requirement = {"library": "numpy"}
    if previous_scope == "other_library":
        old["url"] = "https://pandas.pydata.org/docs/user_guide/merging.html"
        old_requirement = {"library": "pandas"}
    batches.append([old])
    first = tool("official reference", requirement=old_requirement)
    previous = [SearchHit.model_validate(hit) for hit in first["hits"]]
    assert previous
    batches.append([result("numpy.stack", version="1.26")])

    requirement = {"library": "numpy", "symbols": ["numpy.concatenate", "numpy.stack"],
                   "version": "1.26" if previous_scope == "other_version" else None}
    second = tool("numpy.concatenate numpy.stack 1.26", requirement=requirement, previous_hits=previous)

    assert second["diagnostics"]["answerability"] == "partial"
    assert second["diagnostics"]["missing_requirements"] == ["symbol:numpy.concatenate"]
    assert [hit["evidence"]["snapshot"]["title"] for hit in second["hits"]] == ["numpy.stack"]


def test_previous_partial_evidence_survives_when_all_queries_were_attempted(search):
    """Suppressing duplicate API calls still returns the scoped partial evidence already found."""
    tool, batches, seen = search
    requirement = {"library": "numpy", "symbols": ["numpy.concatenate", "numpy.stack"]}
    batches.append([result("numpy.concatenate")])
    first = tool("numpy concatenate stack", requirement=requirement)

    second = tool("numpy concatenate stack", requirement=requirement,
                  previous_hits=[SearchHit.model_validate(hit) for hit in first["hits"]],
                  attempted_queries=first["diagnostics"]["attempted_queries"])

    assert second["diagnostics"]["answerability"] == "partial"
    assert second["hits"] == first["hits"]
    assert second["diagnostics"]["provider_result_count"] == 0
    assert len(seen) == 2


def test_new_search_combines_missing_aspects_with_previous_source_ranges(search):
    """Separate partial source snapshots can jointly cover all requested parameters."""
    tool, batches, seen = search
    requirement = {"library": "numpy", "symbols": ["numpy.reshape"], "aspects": ["order", "copy"]}
    batches.append([result()])
    first = tool("numpy.reshape order copy", requirement=requirement)
    assert first["diagnostics"]["missing_requirements"] == ["aspect:copy"]
    batches.append([result(body="# numpy.reshape\nnumpy.reshape(a, shape, copy=None)\ncopy: Return an independent array when True.\n")])

    second = tool("numpy.reshape copy parameter", requirement=requirement,
                  previous_hits=[SearchHit.model_validate(hit) for hit in first["hits"]],
                  attempted_queries=first["diagnostics"]["attempted_queries"])

    assert second["diagnostics"]["answerability"] == "covered"
    assert second["diagnostics"]["missing_requirements"] == []
    assert second["diagnostics"]["candidate_count"] == 2
    assert first["hits"][0]["evidence"] in [hit["evidence"] for hit in second["hits"]]
    assert len(seen) == 3


@pytest.mark.parametrize(("library", "symbol", "version"), [
    ("numpy", "numpy.reshape", "1.26"),
    ("pandas", "pandas.concat", "2.2"),
])
def test_refinement_quotes_required_symbol_and_version_without_repeating_qualifiers(search, library, symbol, version):
    """An empty search strengthens exact scope while preserving remaining query constraints."""
    tool, batches, seen = search
    query = f"{library} {symbol} {version} API reference preserve empty inputs -experimental"
    payload = tool(query, requirement={"library": library, "symbols": [symbol], "version": version,
                                       "aspects": ["empty inputs"]})

    assert payload["diagnostics"]["answerability"] == "missing"
    assert len(seen) == 2
    refined = seen[1]["query"]
    assert f'"{symbol}"' in refined
    assert f'"{version}"' in refined
    assert "preserve empty inputs -experimental" in refined
    assert refined.casefold().count("api reference") == 1
    assert seen[1]["include_domains"] == seen[0]["include_domains"]
    assert refined != seen[0]["query"]


def test_already_exact_query_is_not_repeated_with_another_reference_qualifier(search):
    """An already scoped exact query has no redundant fallback request."""
    tool, batches, seen = search
    query = 'numpy "numpy.reshape" "1.26" API reference empty inputs -experimental'
    payload = tool(query, requirement={"library": "numpy", "symbols": ["numpy.reshape"], "version": "1.26",
                                       "aspects": ["empty inputs"]})

    assert payload["diagnostics"]["answerability"] == "missing"
    assert payload["diagnostics"]["attempted_queries"] == [query]
    assert len(seen) == 1
