"""Public documentation search ranks and selects evidence for the request."""

import json as json_lib
import sys

import pytest
import requests

from src.core.planner_schema import RetrievalRequirement
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.ranking import DocsAssessmentCache, assess_docs_evidence, select_docs_evidence
from src.infra.tools.docs_search.sections import scan_declarations
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


def python_module_reference(body):
    return {
        "url": "https://docs.python.org/3/library/functools.html",
        "title": "functools — Higher-order functions and operations on callable objects",
        "content": "Functions that act on or return other functions.",
        "raw_content": body,
        "score": 0.9,
    }


@pytest.mark.parametrize("emphasis", ["*maxsize*", "`maxsize`"])
@pytest.mark.parametrize("needs", [["parameters", "explanation"], ["signature"]])
def test_decorator_overload_and_owned_parameter_prose_cover_requirement(search, emphasis, needs):
    """Tavily's Python reference has decorator overloads and prose, not a table."""
    tool, batches, _ = search
    description = f"If {emphasis} is set to `None`, the LRU feature is disabled and the cache can grow without bound."
    body = (
        "# functools\n"
        "functools.cmp_to_key(*func*)[¶](#functools.cmp_to_key \"Link to this definition\")\n"
        "Examples\n>>> sorted(data, key=cmp_to_key(compare))\n\n"
        "@functools.lru\\_cache(*user\\_function*)[¶](#functools.lru_cache \"Link to this definition\")\n\n"
        "@functools.lru\\_cache(*maxsize=128*, *typed=False*)\n"
        ":   Decorator to wrap a function with a memoizing callable.\n\n"
        "    ```\n    @lru_cache\n    def count_vowels(word): return len(word)\n    ```\n\n"
        f"    {description}\n\n"
        "@functools.total\\_ordering[¶](#functools.total_ordering \"Link to this definition\")\n"
        ":   This class decorator supplies the remaining comparison methods.\n"
    )
    batches.append([python_module_reference(body)])
    payload = tool("Python functools.lru_cache maxsize", k=1, requirement=RetrievalRequirement(
        library="Python", symbols=["functools.lru_cache"], aspects=["maxsize"], evidence_needs=needs,
    ))
    assert payload.diagnostics.answerability == "covered"
    assert payload.diagnostics.missing_requirements == []
    assert len(hits(payload)) == 1
    evidence = hits(payload)[0].evidence
    assert evidence.element.text == body
    assert evidence.excerpt == body[evidence.selection.start:evidence.selection.end]
    if "parameters" in needs:
        assert description in evidence.excerpt


@pytest.mark.parametrize(("parameter", "description"), [
    ("maxsize", "*maxsize*가 `None`으로 설정되면, LRU 기능이 비활성화되고 캐시가 제한 없이 커질 수 있습니다."),
    ("limit", "*limit*은 저장할 항목의 최대 개수를 결정합니다."),
    ("mode", "*mode*는 결과를 정렬할 기준을 선택합니다."),
    ("count", "*count*이 0이면 결과에서 마지막 항목이 제거됩니다."),
])
def test_korean_parameter_subject_supplies_owned_behavior(search, parameter, description):
    tool, batches, _ = search
    body = (
        f"@functools.lru\\_cache(*{parameter}=128*, *typed=False*)"
        "[¶](#functools.lru_cache \"Link to this definition\")\n"
        ":   가장 최근의 호출을 저장하는 데코레이터입니다.\n\n"
        f"    {description}\n"
    )
    source = python_module_reference(body)
    source["url"] = "https://docs.python.org/ko/3/library/functools.html"
    batches.append([source])
    payload = tool(f"Python functools.lru_cache {parameter} 설명", k=1, requirement=RetrievalRequirement(
        library="Python", symbols=["functools.lru_cache"], aspects=[parameter],
        evidence_needs=["parameters", "explanation"],
    ))
    assert payload.diagnostics.answerability == "covered"
    assert payload.diagnostics.missing_requirements == []
    evidence = hits(payload)[0].evidence
    assert description in evidence.excerpt
    assert evidence.element.text == body
    assert evidence.excerpt == body[evidence.selection.start:evidence.selection.end]


@pytest.mark.parametrize("description", [
    "*typed*가 참이면 인자 자료형을 구분하며 *maxsize*는 그대로 유지됩니다.",
    "*maxsize*는 버전 3.2에서 추가되었습니다.",
    "*maxsize*는 버전 3.2에 추가되었습니다.",
    "*maxsize*는 선택적 인자로 추가되었습니다.",
    "*maxsize*가 버전 3.2에서 제거되었습니다.",
    "*other*는 결과 항목의 개수를 *maxsize* 이하로 제한합니다.",
])
def test_korean_parameter_prose_rejects_incidental_names_and_lifecycle_metadata(search, description):
    tool, batches, _ = search
    body = (
        "@functools.lru_cache(maxsize=128, typed=False)\n"
        "The decorator caches the result of recent calls.\n\n"
        f"{description}\n"
    )
    source = python_module_reference(body)
    source["url"] = "https://docs.python.org/ko/3/library/functools.html"
    batches.append([source])
    payload = tool("Python functools.lru_cache maxsize 설명", requirement=RetrievalRequirement(
        library="Python", symbols=["functools.lru_cache"], aspects=["maxsize"],
        evidence_needs=["parameters", "explanation"],
    ))
    assert payload.diagnostics.answerability == "partial"
    assert "aspect:maxsize" in payload.diagnostics.missing_requirements


@pytest.mark.parametrize(("description", "expected"), [
    ("*limit*는 버전 3.2에 추가되었습니다.", "partial"),
    ("*limit*는 버전 3.2에서 제거되었습니다.", "partial"),
    ("*limit*는 선택적 인자로 추가되었습니다.", "partial"),
    ("*limit*가 0이면 결과에서 마지막 항목이 제거됩니다.", "covered"),
])
def test_korean_lifecycle_metadata_does_not_supply_definition_explanation(search, description, expected):
    tool, batches, _ = search
    body = f"functools.memoize(limit=128)\n\n{description}\n"
    source = python_module_reference(body)
    source["url"] = "https://docs.python.org/ko/3/library/functools.html"
    batches.append([source])
    payload = tool("functools.memoize 설명", requirement=RetrievalRequirement(
        library="Python", symbols=["functools.memoize"], evidence_needs=["explanation"],
    ))
    assert payload.diagnostics.answerability == expected
    assert payload.diagnostics.missing_requirements == (["need:explanation"] if expected == "partial" else [])


@pytest.mark.parametrize("body", [
    "@functools.lru_cache(maxsize=128, typed=False)\n",
    "@functools.lru_cache(maxsize=128, typed=False)\n"
    "The decorator caches the result of recent calls.\n\n"
    "If typed is true, calls are distinguished by type while maxsize remains unchanged.\n",
    "@functools.lru_cache(maxsize=128, typed=False)\n"
    "The decorator caches the result of recent calls.\n\n"
    "Changed in version 3.2: The maxsize parameter was added.\n",
    "@functools.lru_cache(maxsize=128, typed=False)\n"
    "Parameters\ntyped: Controls whether argument types distinguish cache entries.\n"
    "If maxsize is None, this option still distinguishes argument types.\n",
    "@functools.lru_cache(maxsize=128, typed=False)\n"
    "The decorator caches the result of recent calls.\n\n"
    "@functools.total_ordering[¶](#functools.total_ordering \"Link to this definition\")\n"
    "If maxsize is None, this unrelated decorator uses an unlimited cache.\n",
])
def test_parameter_prose_requires_own_behavior_not_signatures_or_incidental_mentions(search, body):
    tool, batches, _ = search
    batches.append([python_module_reference(body)])
    payload = tool("Python functools.lru_cache maxsize", requirement=RetrievalRequirement(
        library="Python", symbols=["functools.lru_cache"], aspects=["maxsize"],
        evidence_needs=["parameters", "explanation"],
    ))
    assert payload.diagnostics.answerability == "partial"
    assert "aspect:maxsize" in payload.diagnostics.missing_requirements


def test_parameter_prose_uses_declaration_for_identity_but_selected_text_for_behavior():
    body = (
        "@functools.lru_cache(maxsize=128, typed=False)\n\n"
        "If *maxsize* is set to None, the cache can grow without bound.\n\n"
        "If *typed* is true, calls with different argument types are cached separately.\n"
    )
    source = python_module_reference(body)
    requirement = RetrievalRequirement(
        library="Python", symbols=["functools.lru_cache"], aspects=["maxsize"],
        evidence_needs=["parameters", "explanation"],
    )
    description = _source_evidence(source, start=body.index("If *maxsize*"), end=body.index("If *typed*"))
    unrelated = _source_evidence(source, start=body.index("If *typed*"))
    cropped = _source_evidence(source, start=body.index("If *maxsize*"), end=body.index("without bound"))
    assert assess_docs_evidence("functools.lru_cache maxsize", [description], requirement).answerability == "covered"
    assert "aspect:maxsize" in assess_docs_evidence("functools.lru_cache maxsize", [unrelated], requirement).missing
    assert "aspect:maxsize" in assess_docs_evidence("functools.lru_cache maxsize", [cropped], requirement).missing


@pytest.mark.parametrize("parameter", ["__name__", "_private", "max_size", "_private_"])
def test_code_identifier_underscores_remain_part_of_parameter_identity(search, parameter):
    tool, batches, _ = search
    body = (
        f"functools.inspect(a, {parameter})\n\n"
        f"If `{parameter}` is None, the function returns the original object.\n"
    )
    batches.append([python_module_reference(body)])
    payload = tool(f"functools.inspect {parameter}", requirement=RetrievalRequirement(
        library="Python", symbols=["functools.inspect"], aspects=[parameter],
        evidence_needs=["parameters", "explanation"],
    ))
    assert payload.diagnostics.answerability == "covered"
    assert hits(payload)[0].evidence.element.text == body


@pytest.mark.parametrize("symbol", ["functools._cache_", "functools.cache"])
@pytest.mark.parametrize("markup", ["`functools._cache_`", r"functools.\_cache\_"])
def test_code_and_escaped_underscores_do_not_change_api_identity(search, symbol, markup):
    tool, batches, _ = search
    body = f"# {markup}\n`functools._cache_(maxsize=None)`\nIf maxsize is None, the cache grows without bound.\n"
    batches.append([python_module_reference(body)])
    payload = tool(f"{symbol} maxsize", requirement=RetrievalRequirement(
        library="Python", symbols=[symbol], aspects=["maxsize"], evidence_needs=["explanation"],
    ))
    assert payload.diagnostics.answerability == ("covered" if symbol == "functools._cache_" else "missing")


def _source_evidence(source, *, start=0, end=None):
    body = source["raw_content"]
    snapshot = build_snapshot(
        source_uri=source["url"], title=source["title"], media_type="text/markdown",
        source_type="official", content=body, parser="test", parser_version="1",
    )
    return build_evidence(
        snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=body),
        start=start, end=end,
    )


@pytest.mark.parametrize("fenced", [False, True], ids=["plain", "signature-fence"])
def test_multiline_declaration_supplies_signature_and_owned_parameter_prose(search, fenced):
    tool, batches, _ = search
    declaration = "numpy.mean(\n    a,\n    axis=None,\n)\n"
    body = "# numpy.mean\n" + (
        "Signature\n```python\n" + declaration + "```\n" if fenced else declaration
    ) + "\nIf axis is None, all input dimensions are reduced.\n"
    batches.append([reference("numpy.mean", body=body)])
    payload = tool("numpy.mean axis", requirement=RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], aspects=["axis"],
        evidence_needs=["signature", "parameters", "explanation"],
    ))
    assert payload.diagnostics.answerability == "covered"
    assert payload.diagnostics.missing_requirements == []
    evidence = hits(payload)[0].evidence
    assert evidence.element.text == body
    assert evidence.excerpt == body[evidence.selection.start:evidence.selection.end]


@pytest.mark.parametrize("nested_name", ["dict", "mean", "numpy.mean"])
def test_cropped_nested_call_does_not_supply_the_outer_api_signature(nested_name):
    body = (
        "# numpy.mean\nnumpy.mean(\n    a,\n    config=\n"
        f"        {nested_name}(mode='fast')\n)\nCompute the arithmetic mean of the input.\n"
    )
    nested_start = body.index(f"        {nested_name}")
    evidence = _source_evidence(reference("numpy.mean", body=body),
                                start=nested_start, end=nested_start + len(f"        {nested_name}(mode='fast')\n"))
    assessment = assess_docs_evidence("numpy.mean signature", [evidence], RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], evidence_needs=["signature"],
    ))
    assert "need:signature" in assessment.missing


@pytest.mark.parametrize("decoration", [
    "  ",
    '[#](#numpy.mean "Link to this definition")',
    '[[source]](https://example.com/source.py)[¶](#numpy.mean "Permalink to this definition")',
])
def test_selected_signature_does_not_require_its_trailing_metadata(decoration):
    declaration = "numpy.mean(\n    a,\n    axis=None,\n)"
    body = f"# numpy.mean\n{declaration}{decoration}\nCompute the arithmetic mean of the input.\n"
    start = body.index(declaration)
    evidence = _source_evidence(reference("numpy.mean", body=body), start=start, end=start + len(declaration))
    assessment = assess_docs_evidence("numpy.mean signature", [evidence], RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], evidence_needs=["signature"],
    ))
    assert assessment.answerability == "covered"


@pytest.mark.parametrize("annotation", ["AxisType | int | None", "int | tuple[int, ...] | None"])
@pytest.mark.parametrize("description", ["", "If axis is None, all input dimensions are reduced.\n"])
def test_multiline_annotation_is_not_parameter_behavior_or_definition_explanation(annotation, description):
    body = (
        "# numpy.mean\nSignature\nnumpy.mean(\n    a,\n"
        f"    axis: {annotation} = None,\n)\n{description}"
    )
    evidence = _source_evidence(reference("numpy.mean", body=body))
    requirement = RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], aspects=["axis"],
        evidence_needs=["parameters", "explanation"],
    )
    assessment = assess_docs_evidence("numpy.mean axis", [evidence], requirement)
    assert assessment.answerability == ("covered" if description else "partial")
    if not description:
        assert assessment.missing == ["aspect:axis", "need:explanation", "need:parameters"]


@pytest.mark.parametrize(("signature", "parameter"), [
    ('numpy.mean(a, config=dict(mode="x", timeout=30))', "timeout"),
    ('numpy.mean(a, config={"first": 1, "second": dict(timeout=30)})', "timeout"),
    ('numpy.mean(a, text="first, timeout=30")', "timeout"),
    ("numpy.mean(a, config=lambda mode, timeout: True)", "timeout"),
    ("numpy.mean(a, config=lambda mode=lambda x, y: x + y, timeout=30: mode)", "timeout"),
    ("numpy.mean(a) -> tuple(int, result)", "result"),
    ("numpy.mean(a, mode: tuple[int, result] = None)", "result"),
])
def test_nested_default_and_annotation_names_are_not_declared_parameters(signature, parameter):
    # Synthetic signatures isolate grammar boundaries, independent of NumPy's API.
    body = f"# numpy.mean\n{signature}\n{parameter} controls which records are returned to the caller.\n"
    evidence = _source_evidence(reference("numpy.mean", body=body))
    assessment = assess_docs_evidence(f"numpy.mean {parameter}", [evidence], RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], aspects=[parameter],
        evidence_needs=["parameters", "explanation"],
    ))
    assert assessment.answerability == "partial"
    assert f"aspect:{parameter}" in assessment.missing
    assert "need:parameters" in assessment.missing


@pytest.mark.parametrize("signature", [
    'numpy.mean(a, config=dict(mode="x", timeout=30), axis=None)',
    'numpy.mean(a, text="literal ), axis=fake", axis=None)',
    'numpy.mean(a, config={"first": (1, 2), "second": [3, 4]}, axis=None)',
    'numpy.mean(a, config: tuple[int, str] = (1, "x"), axis=None) -> tuple(int, result)',
    "numpy.mean(a, config=lambda mode, timeout: True, axis=None)",
    "numpy.mean(a, config=lambda mode=lambda x, y: x + y, timeout=30: mode, axis=None)",
    "numpy.mean(a, config=_lambda, axis=None)",
    "numpy.mean(a, config=lambda_factory, axis=None)",
    'numpy.mean(a, config="lambda mode, timeout: None", axis=None)',
    "numpy.mean(a[, axis])",
    "numpy.mean(a[, config[, axis]])",
])
def test_top_level_parameters_after_nested_values_retain_their_behavior(signature):
    body = f"# numpy.mean\n{signature}\nIf axis is None, all input dimensions are reduced.\n"
    evidence = _source_evidence(reference("numpy.mean", body=body))
    assessment = assess_docs_evidence("numpy.mean axis", [evidence], RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], aspects=["axis"],
        evidence_needs=["signature", "parameters", "explanation"],
    ))
    assert assessment.answerability == "covered"


@pytest.mark.parametrize("label", ["Note", "Warning", "Result"])
@pytest.mark.parametrize("with_parameter_prose", [False, True])
def test_signature_metadata_is_not_a_parameter_or_a_barrier_to_its_prose(label, with_parameter_prose):
    body = f"# numpy.mean\nnumpy.mean(a, axis=None)\n{label}: This function accepts array inputs.\n"
    if with_parameter_prose:
        body += "axis controls which dimension is reduced by this function.\n"
    evidence = _source_evidence(reference("numpy.mean", body=body))
    requirement = RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], evidence_needs=["parameters", "explanation"],
    )
    assessment = assess_docs_evidence("numpy.mean parameters", [evidence], requirement)
    assert assessment.answerability == ("covered" if with_parameter_prose else "partial")
    assert ("need:parameters" in assessment.missing) is not with_parameter_prose
    unrelated = assess_docs_evidence("numpy.mean metadata", [evidence], requirement.model_copy(update={"aspects": [label]}))
    assert f"aspect:{label}" in unrelated.missing
    if with_parameter_prose:
        own = assess_docs_evidence("numpy.mean axis", [evidence], requirement.model_copy(update={"aspects": ["axis"]}))
        assert own.answerability == "covered"


def test_cached_assessment_rechecks_ranges_requirements_and_source_revisions():
    body = (
        "# numpy.mean\nnumpy.mean(a, axis=None)\nCompute the arithmetic mean of array elements.\n\n"
        "Parameters\naxis: Axis along which means are computed.\n\n"
        "Examples\n>>> numpy.mean([1, 2])\n1.5\n"
    )
    source = reference("numpy.mean", body=body)
    complete = _source_evidence(source)
    parameters = _source_evidence(source, start=body.index("Parameters"), end=body.index("Examples"))
    examples = _source_evidence(source, start=body.index("Examples"))
    revised = _source_evidence(reference("numpy.mean", body=body.replace("axis", "dtype")))
    requirement = RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], aspects=["axis"], evidence_needs=["parameters", "examples"],
    )
    cache = DocsAssessmentCache()
    cases = [
        ([complete], requirement, []),
        ([parameters], requirement, ["need:examples"]),
        ([parameters, examples], requirement, []),
        ([examples, parameters], requirement, []),
        ([complete], requirement.model_copy(update={"aspects": ["dtype"]}), ["aspect:dtype"]),
        ([revised], requirement, ["aspect:axis"]),
        ([complete], requirement, []),
    ]
    for selected, requested, missing in cases:
        actual = assess_docs_evidence("numpy.mean parameters examples", selected, requested, cache=cache)
        assert actual == assess_docs_evidence("numpy.mean parameters examples", selected, requested)
        assert actual.missing == missing


def test_cached_topic_assessment_rechecks_focus_when_another_api_enters_packet():
    incidental = _source_evidence(reference("numpy.var", body=(
        "# numpy.var\nnumpy.var(a, mean=None)\n\nParameters\n"
        "mean: Supplies the mean value used to compute the variance.\n"
    )))
    target_heading = _source_evidence(reference("numpy.mean", body="# numpy.mean\n"))
    requirement = RetrievalRequirement(library="numpy", evidence_needs=["parameters"])
    cache = DocsAssessmentCache()
    for selected, missing in (([incidental], []), ([incidental, target_heading], ["need:parameters", "topic"]),
                              ([incidental], [])):
        actual = assess_docs_evidence("mean", selected, requirement, cache=cache)
        assert actual == assess_docs_evidence("mean", selected, requirement)
        assert actual.missing == missing


