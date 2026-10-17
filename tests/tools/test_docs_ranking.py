"""Public documentation search ranks and selects evidence for the request."""

import json as json_lib

import pytest
import requests

from src.core.planner_schema import RetrievalRequirement
from src.core.evidence import SearchHit
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool


@pytest.fixture
def search(monkeypatch):
    """Keep retrieval logic real and replace only the external HTTP boundary."""
    batches = []
    requests_seen = []

    def post(url, *, json: dict, **kwargs):
        requests_seen.append(json)
        response = requests.Response()
        response.status_code = 200
        response._content = json_lib.dumps(
            {"results": batches.pop(0) if batches else []}
        ).encode()
        return response

    monkeypatch.setattr(requests, "post", post)
    tool = build_docs_search_tool(
        AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test")
    )
    yield tool, batches, requests_seen


def reference(symbol, *, parameter="axis", body=None, score=0.4, version="stable", suffix=""):
    library = symbol.split(".")[0]
    roots = {
        "matplotlib": f"https://matplotlib.org/{version}/api/_as_gen/",
        "numpy": f"https://numpy.org/doc/{version}/reference/generated/",
        "pandas": "https://pandas.pydata.org/docs/reference/api/",
    }
    return {
        "url": roots[library] + symbol + ".html" + suffix,
        "title": symbol,
        "content": f"{symbol} documentation.",
        "raw_content": body or (
            f"# {symbol}\n{symbol}(data, {parameter}=None)\n"
            f"Apply {symbol} to the supplied data.\n\nParameters\n"
            f"{parameter}: Controls how the supplied data is processed.\n"
        ),
        "score": score,
    }


def hits(payload):
    return [SearchHit.model_validate(item) for item in payload.hits]


@pytest.mark.parametrize("explicit", [False, True], ids=["short-query", "explicit-symbol"])
def test_target_api_beats_unrelated_rich_reference(search, explicit):
    """The requested API outranks an unrelated source with a higher provider score."""
    tool, batches, _ = search
    library, symbol, other, parameter = "numpy", "numpy.mean", "numpy.reshape", "axis"
    unrelated = reference(other, score=0.99)
    unrelated["raw_content"] += "\nReturns\nresult: The computed result.\nExamples\n>>> data = [1, 2]\n"
    target = reference(symbol, parameter=parameter, score=0.3)
    batches.append([unrelated, target])
    requirement = RetrievalRequirement(
        library=library, symbols=[symbol] if explicit else [], evidence_needs=["parameters"],
    )
    payload = tool(f"{library} {symbol.rsplit('.', 1)[-1]} 옵션 설명", k=1, requirement=requirement)
    selected = hits(payload)
    assert selected[0].evidence.snapshot.source_uri == target["url"]
    assert f"{parameter}: Controls" in selected[0].evidence.excerpt
    assert payload.diagnostics.answerability == "covered"


def test_library_match_alone_does_not_cover_a_topic(search):
    tool, batches, _ = search
    batches.append([reference("numpy.reshape", parameter="shape")])
    payload = tool("numpy broadcasting", requirement=RetrievalRequirement(library='numpy'), k=1)
    assert payload.diagnostics.answerability == "missing"
    assert hits(payload) == []


def test_complementary_aspects_are_selected_before_repeated_coverage(search):
    tool, batches, _ = search
    order = reference("numpy.reshape", parameter="order", score=0.99, suffix="#order")
    duplicate = reference("numpy.reshape", parameter="order", score=0.98, suffix="#order-notes")
    copy = reference("numpy.reshape", parameter="copy", score=0.2, suffix="#copy")
    batches.append([order, duplicate, copy])
    payload = tool("numpy.reshape order copy", k=2, requirement=RetrievalRequirement(library='numpy', symbols=['numpy.reshape'], aspects=['order', 'copy'], evidence_needs=['parameters']))
    selected = hits(payload)
    assert payload.diagnostics.answerability == "covered"
    assert len(selected) <= 2
    text = "\n".join(hit.evidence.excerpt for hit in selected)
    assert "order: Controls" in text
    assert "copy: Controls" in text


def test_parameter_of_another_api_on_same_page_does_not_cover_target(search):
    tool, batches, _ = search
    source = reference("numpy.mean", body=(
        "# numpy.mean\nnumpy.mean(a, dtype=None)\nCompute an arithmetic mean.\n"
        "Parameters\ndtype: The type used to compute the mean.\n\n"
        "# numpy.sum\nnumpy.sum(a, axis=None)\nCompute a sum.\n"
        "Parameters\naxis: Axis along which the sum is computed.\n"
    ))
    batches.append([source])
    payload = tool("numpy.mean axis", k=2, requirement=RetrievalRequirement(library='numpy', symbols=['numpy.mean'], aspects=['axis'], evidence_needs=['parameters']))
    assert payload.diagnostics.answerability == "partial"
    assert hits(payload)
    assert any("axis" in item for item in payload.diagnostics.missing_requirements)


@pytest.mark.parametrize("source_version,expected", [("stable", "partial"), ("2.4", "missing")])
def test_unknown_and_conflicting_versions_are_distinct(search, source_version, expected):
    tool, batches, _ = search
    batches.append([reference("numpy.mean", version=source_version)])
    payload = tool("numpy.mean 1.26 parameters", k=1, requirement=RetrievalRequirement(library='numpy', symbols=['numpy.mean'], version='1.26', evidence_needs=['parameters']))
    assert payload.diagnostics.answerability == expected
    assert any("version" in item for item in payload.diagnostics.missing_requirements)
    assert bool(hits(payload)) is (expected == "partial")


def test_korean_topic_cannot_be_replaced_by_unrelated_api_metadata(search):
    """Korean subject words still constrain a query without an explicit API."""
    tool, batches, _ = search
    target = {
        "url": "https://numpy.org/doc/stable/user/basics.broadcasting.html",
        "title": "브로드캐스팅 규칙",
        "content": "브로드캐스팅은 크기가 다른 배열 사이의 연산을 지원합니다. 각 차원의 크기가 같거나 하나가 1이면 호환됩니다.",
        "score": 0.3,
    }
    unrelated = reference("numpy.mean", parameter="axis", score=0.99)
    batches.append([unrelated, target])
    payload = tool("numpy 브로드캐스팅 규칙 설명", requirement=RetrievalRequirement(library='numpy'), k=1)
    assert payload.diagnostics.answerability == "covered"
    assert hits(payload)[0].evidence.snapshot.source_uri == target["url"]


def test_reference_page_examples_support_usage_but_not_signature(search):
    """A generated-reference URL cannot turn a concrete call into a signature."""
    tool, batches, _ = search
    source = {
        "url": "https://numpy.org/doc/stable/reference/generated/numpy.mean.html",
        "title": "Mean examples",
        "content": "Calculate the mean of two integers with numpy.mean.",
        "raw_content": "# Examples\n>>> import numpy\n>>> numpy.mean([1, 2])\n1.5\n",
        "score": 0.9,
    }
    batches.append([source])
    example = tool("numpy.mean examples", k=1, requirement=RetrievalRequirement(library='numpy', symbols=['numpy.mean'], evidence_needs=['examples']))
    assert example.diagnostics.answerability == "covered"
    assert "numpy.mean([1, 2])" in hits(example)[0].evidence.excerpt

    batches.append([source])
    signature = tool("numpy.mean signature", k=1, requirement=RetrievalRequirement(library='numpy', symbols=['numpy.mean'], evidence_needs=['signature']))
    assert signature.diagnostics.answerability != "covered"
    assert "need:signature" in signature.diagnostics.missing_requirements
