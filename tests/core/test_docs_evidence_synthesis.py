"""Documentation requirements survive the actual model evidence budget."""

import json
from pathlib import Path
import subprocess
import sys

import requests
import pytest

from src.core.contracts import PlannerState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.evidence import SearchHit
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs


def _retrieve(monkeypatch, body, *, need, symbol="numpy.mean", aspects=None, library="numpy"):
    """Only HTTP is fake; search selection and synthesis preparation are real."""
    document_symbol = symbol.replace("np.", "numpy.")
    source = {
        "url": (f"https://pandas.pydata.org/docs/reference/api/{document_symbol}.html" if library == "pandas"
                else f"https://numpy.org/doc/stable/reference/generated/{document_symbol}.html"),
        "title": document_symbol, "content": body, "raw_content": body, "score": 0.9,
    }

    def post(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"results": [source]}).encode()
        return response

    monkeypatch.setattr(requests, "post", post)
    needs = [need] if isinstance(need, str) else need
    task = RetrievalTask(
        route="docs", query=f"{symbol} {' '.join(needs)}", k=1,
        requirement={"library": library, "symbols": [symbol], "evidence_needs": needs, "aspects": aspects or []},
    )
    result = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test"))(
        task.query, requirement=task.requirement, k=task.k,
    )
    assert result.diagnostics.answerability == "covered"
    hits = [SearchHit.model_validate(hit).model_copy(update={"requirement_id": task.requirement_id})
            for hit in result.hits]
    return task, hits


def _prepare(task, hits, *, excerpt_chars=1800, max_items=8):
    state = build_graph_state_input(
        user_input=task.query, planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        request_contract=RequestContract(),
        retrieval={"hit_log": [hit.model_dump(mode="json") for hit in hits]},
    )
    prepared = prepare_synthesis_inputs(
        state=state, context=build_synthesis_context(state=state),
        budget=RetrievedEvidenceBudget(excerpt_chars, 6000, max_items), max_turns=6,
    )
    requirement_message = next(str(message.content) for message in prepared.model_messages
                               if str(message.content).startswith("[Retrieval Requirements]"))
    records = json.loads(requirement_message[requirement_message.index("\n[") + 1:])
    return prepared, records[0]["coverage"]


@pytest.mark.parametrize("excerpt_chars", [120, 1800])
def test_multiline_signature_survives_budget_beside_a_long_unbroken_description(monkeypatch, excerpt_chars):
    signature = (
        "pandas.read_csv(\n"
        "    filepath_or_buffer,\n"
        "    sep=None,\n"
        "    dtype=None,\n"
        "    usecols=None,\n"
        ")"
    )
    description = "dtype: Selects the desired type for output columns."
    body = (
        "# pandas.read_csv\n" + signature + "\nRead a CSV file into a DataFrame. "
        + "Additional details about parsing input files and handling output types. " * 35
        + "\nParameters\nsep: Selects the input delimiter used to read input files.\n"
        + description + "\n"
    )
    task, hits = _retrieve(monkeypatch, body, need=["signature", "explanation"],
                           symbol="pandas.read_csv", library="pandas")

    prepared, coverage = _prepare(task, hits, excerpt_chars=excerpt_chars)

    assert coverage["missing_requirements"] == []
    assert prepared.missing_requirement_ids == []
    assert any(signature in item.excerpt for item in prepared.evidence_packet)
    assert any(description in item.excerpt for item in prepared.evidence_packet)
    _assert_budget_case_ranges(
        prepared.evidence_packet, [hit.evidence for hit in hits], RetrievedEvidenceBudget(excerpt_chars, 6000, 8),
    )


@pytest.mark.parametrize(("clipped_start", "clipped_end", "excerpt_chars"), [(1, 0, 1800), (0, 1, 1800), (0, 0, 40)])
def test_signature_outside_retrieved_range_or_budget_stays_missing(clipped_start, clipped_end, excerpt_chars):
    from src.core.evidence import build_evidence
    from src.runtime.nodes.synthesis.evidence_selection import requirement_coverage, select_evidence_packet

    signature = "pandas.read_csv(\n    filepath_or_buffer,\n    sep=None,\n    dtype=None,\n)"
    body = "# pandas.read_csv\n" + signature + "\n"
    source = _budget_case_source(body, "bounded-signature")
    original = build_evidence(
        snapshot=source.snapshot, element=source.element,
        start=body.index(signature) + clipped_start, end=body.index(signature) + len(signature) - clipped_end,
    )
    task = RetrievalTask(
        route="docs", query="pandas.read_csv signature", k=1,
        requirement={"library": "pandas", "symbols": ["pandas.read_csv"], "evidence_needs": ["signature"]},
    )
    budget = RetrievedEvidenceBudget(excerpt_chars, 6000, 8)

    packet, ids = select_evidence_packet(
        [original], budget=budget, query=task.query, requirements_by_evidence={original.id: [task]},
    )

    _assert_budget_case_ranges(packet, [original], budget)
    assert "need:signature" in requirement_coverage(task, packet, ids)["missing_requirements"]
    assert all(signature not in item.excerpt for item in packet)


def test_parameter_aspect_does_not_displace_a_separate_required_example(monkeypatch):
    """A covered parameter cannot repeatedly defeat the still-missing example passage."""
    description = (
        "order: C reads and writes the last axis first, while F reads and writes the first axis first. "
        "A follows Fortran order for a Fortran contiguous array and C order otherwise. "
        + "The index order controls reading and writing values without changing the original array data. " * 6
        + "\n"
    )
    examples = "Examples\n>>> numpy.reshape([1, 2, 3, 4], (2, 2))\narray([[1, 2], [3, 4]])\n"
    body = (
        "# numpy.reshape\nnumpy.reshape(a, shape, order='C')\n\nParameters\n"
        + description + "\nNotes\n" + "Additional background about storage and memory layout. " * 25
        + "\n\n" + examples
    )
    assert len(body) > 1800
    task, hits = _retrieve(monkeypatch, body, need=["examples", "explanation", "parameters"],
                           symbol="numpy.reshape", aspects=["order"])

    prepared, coverage = _prepare(task, hits)

    assert coverage["missing_aspects"] == []
    assert coverage["missing_needs"] == []
    assert prepared.missing_requirement_ids == []
    assert any(description.strip() in item.excerpt for item in prepared.evidence_packet)
    assert any(examples.strip() in item.excerpt for item in prepared.evidence_packet)
    assert len(prepared.evidence_packet) >= 2
    assert all(len(item.excerpt) <= 1800 for item in prepared.evidence_packet)
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= 6000
    for selected in prepared.evidence_packet:
        assert any(
            original.evidence.snapshot == selected.snapshot
            and original.evidence.element == selected.element
            and original.evidence.selection.start <= selected.selection.start
            and selected.selection.end <= original.evidence.selection.end
            for original in hits
        )
        assert selected.excerpt == body[selected.selection.start:selected.selection.end]


@pytest.mark.parametrize(("introduction", "condition"), [
    (
        "Decorator to wrap a function with a memoizing callable that stores up to the *limit* most recent calls.",
        "If *limit* is set to `None`, the cache can grow without bound.",
    ),
    (
        "가장 최근의 *limit* 호출까지 저장하는 메모하는 콜러블로 함수를 감싸는 데코레이터입니다.",
        "*limit*가 `None`으로 설정되면, 캐시가 제한 없이 커질 수 있습니다.",
    ),
])
@pytest.mark.parametrize("max_items", [1, 8])
def test_parameter_behavior_budget_keeps_definition_context_and_later_condition(
    monkeypatch, introduction, condition, max_items,
):
    """A special-value condition alone cannot replace the parameter's main meaning."""
    background = "The cache keeps its underlying data structure coherent during concurrent updates. "
    body = (
        "# numpy.memoize\n@numpy.memoize(limit=128, typed=False)\n"
        f":   {introduction}\n\n"
        + background * 21 + "\n\n"
        + condition + "\n\n"
        + "The cache_info function reports the total number of calls and cache hits.\n"
    )
    assert 1800 < len(body) <= 3200
    task, hits = _retrieve(monkeypatch, body, need=["parameters", "explanation"],
                           symbol="numpy.memoize", aspects=["limit"])
    assert any(introduction in hit.evidence.excerpt and condition in hit.evidence.excerpt for hit in hits)

    prepared, coverage = _prepare(task, hits, max_items=max_items)

    assert coverage["missing_requirements"] == []
    assert prepared.missing_requirement_ids == []
    if max_items > 1:
        assert any(introduction in item.excerpt for item in prepared.evidence_packet)
    assert any(condition in item.excerpt for item in prepared.evidence_packet)
    assert len(prepared.evidence_packet) <= max_items
    assert all(len(item.excerpt) <= 1800 for item in prepared.evidence_packet)
    assert sum(len(item.excerpt) for item in prepared.evidence_packet) <= 6000
    for selected in prepared.evidence_packet:
        assert selected.excerpt == body[selected.selection.start:selected.selection.end]
        assert any(
            original.evidence.snapshot == selected.snapshot
            and original.evidence.element == selected.element
            and original.evidence.selection.start <= selected.selection.start
            and selected.selection.end <= original.evidence.selection.end
            for original in hits
        )


def test_definition_context_does_not_import_another_apis_introduction(monkeypatch):
    unrelated = "This unrelated callable stores up to the *limit* most recent calls."
    condition = "If *limit* is set to `None`, the cache can grow without bound."
    body = (
        "# numpy.other\n@numpy.other(limit=128)\n"
        + unrelated + "\n\n"
        + "This background describes another callable and its implementation. " * 27 + "\n\n"
        + "# numpy.memoize\n@numpy.memoize(limit=128)\n"
        + condition + "\n"
    )
    assert 1800 < len(body) <= 3200
    task, hits = _retrieve(monkeypatch, body, need=["parameters", "explanation"],
                           symbol="numpy.memoize", aspects=["limit"])

    prepared, coverage = _prepare(task, hits)

    assert coverage["missing_requirements"] == []
    assert any(condition in item.excerpt for item in prepared.evidence_packet)
    assert all(unrelated not in item.excerpt for item in prepared.evidence_packet)


@pytest.mark.parametrize(("excerpt_chars", "total_chars"), [(200, 700), (1800, 6000)])
def test_definition_context_does_not_displace_additional_source_parameter_behavior(excerpt_chars, total_chars):
    from src.core.documents import DocumentElement, build_snapshot
    from src.core.evidence import build_evidence
    from src.runtime.nodes.synthesis.evidence_selection import requirement_coverage, select_evidence_packet

    task = RetrievalTask(
        route="docs", query="numpy.mean axis behavior for a tuple of axes", k=2,
        requirement={"library": "numpy", "symbols": ["numpy.mean"], "aspects": ["axis"],
                     "evidence_needs": ["parameters", "explanation"]},
    )
    introduction = "Compute the arithmetic mean along the specified axis."
    tuple_behavior = "axis: If this is a tuple of integers, a mean is performed over multiple axes instead of a single axis."
    bodies = [
        "# numpy.mean\n\nParameters\naxis: Axis along which the means are computed.\n",
        "# numpy.mean\n\n" + introduction + "\n\n"
        + "This paragraph discusses general use of the function and its return values. " * 28
        + "\n\nParameters\n" + tuple_behavior + "\n",
    ]
    originals = []
    for version, body in zip(("stable", "2.3"), bodies):
        snapshot = build_snapshot(
            source_uri=f"https://numpy.org/doc/{version}/reference/generated/numpy.mean.html",
            title="numpy.mean", media_type="text/markdown", source_type="official",
            content=body, parser="test", parser_version="1",
        )
        originals.append(build_evidence(
            snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=body),
        ))
    budget = RetrievedEvidenceBudget(excerpt_chars, total_chars, 8)

    packet, ids = select_evidence_packet(
        originals, budget=budget, query=task.query,
        requirements_by_evidence={item.id: [task] for item in originals},
    )

    _assert_budget_case_ranges(packet, originals, budget)
    assert requirement_coverage(task, packet, ids)["missing_requirements"] == []
    assert any(tuple_behavior in item.excerpt for item in packet)
    assert any(introduction in item.excerpt for item in packet)
    assert all(ids[item.id] == [task.requirement_id] for item in packet)


@pytest.mark.parametrize("symbol", ["numpy.mean", "np.mean"])
@pytest.mark.parametrize("excerpt_chars", [200, 1800])
def test_definition_context_uses_the_same_symbol_aliases_as_coverage(monkeypatch, symbol, excerpt_chars):
    introduction = "Compute the arithmetic mean along the specified axis."
    condition = "If axis is None, all dimensions are reduced to a scalar result."
    body = (
        "# numpy.mean\nnumpy.mean(a, axis=None)\n\n" + introduction + "\n\n"
        + "The function describes general arithmetic operations on input arrays. " * 28
        + "\n\n" + condition + "\n"
    )
    task, hits = _retrieve(monkeypatch, body, need=["parameters", "explanation"],
                           symbol=symbol, aspects=["axis"])

    prepared, coverage = _prepare(task, hits, excerpt_chars=excerpt_chars)

    assert coverage["missing_requirements"] == []
    assert prepared.missing_requirement_ids == []
    assert any(introduction in item.excerpt for item in prepared.evidence_packet)
    assert any(condition in item.excerpt for item in prepared.evidence_packet)
    _assert_budget_case_ranges(
        prepared.evidence_packet, [hit.evidence for hit in hits], RetrievedEvidenceBudget(excerpt_chars, 6000, 8),
    )


def test_long_parameter_description_keeps_complete_coverage_within_normal_budget():
    """A long source line must not remain split while its complete text fits.

    The live read_csv failure combined two official versions, duplicate
    parameter ranges, and a multi-sentence usecols description on one line.
    Other parameters occupied earlier ranges; optional context consumed the
    remaining allowance before the incomplete usecols range was repaired.
    """
    from src.core.documents import DocumentElement, build_snapshot
    from src.core.evidence import build_evidence
    from src.runtime.nodes.synthesis.evidence_selection import (
        contains_evidence_range, requirement_coverage, select_evidence_packet,
    )

    aspects = ["sep", "dtype", "parse_dates", "na_values", "usecols", "converters", "chunksize", "iterator"]
    task = RetrievalTask(
        requirement_id="csv_parameters", route="docs", query="pandas.read_csv parameters and examples", k=6,
        requirement={"library": "pandas", "symbols": ["pandas.read_csv"], "aspects": aspects,
                     "evidence_needs": ["parameters", "explanation", "examples"]},
    )

    def parameter(name, sentences):
        return (
            f"        **{name}**str, optional\n        :   Controls how {name} reads input values. "
            + "This setting changes how selected values are read from the input file. " * sentences
            + "\n\n"
        )

    descriptions = {name: parameter(name, count) for name, count in zip(aspects, [5, 3, 6, 4, 8, 1, 2, 1])}
    context = parameter("other_parameter", 20)
    example = "Examples\n>>> pandas.read_csv('input.csv')\n\n"
    fragments = [
        "# pandas.read_csv\npandas.read_csv(filepath_or_buffer, sep=',')\n\nParameters\n"
        + descriptions["sep"] + context,
        context + descriptions["usecols"] + descriptions["dtype"] + descriptions["converters"],
        context + descriptions["na_values"] + descriptions["parse_dates"],
        context + descriptions["iterator"] + descriptions["chunksize"],
        example,
    ]
    body = "".join(fragments)
    versions = []
    for version in ("2.2", "2.3"):
        snapshot = build_snapshot(
            source_uri=f"https://pandas.pydata.org/pandas-docs/version/{version}/reference/api/pandas.read_csv.html",
            title="pandas.read_csv", media_type="text/markdown", source_type="official",
            content=body, parser="test", parser_version="1",
        )
        element = DocumentElement(element_id="body", kind="paragraph", text=body)
        ranges = []
        start = 0
        for fragment in fragments:
            ranges.append(build_evidence(snapshot=snapshot, element=element, start=start, end=start + len(fragment)))
            start += len(fragment)
        versions.append(ranges)
    originals = [versions[0][3], versions[1][0], versions[1][1], versions[0][1], versions[0][2], versions[0][4]]
    original_ids = {item.id: [task.requirement_id] for item in originals}
    assert requirement_coverage(task, originals, original_ids)["missing_requirements"] == []

    # Complete support is feasible without increasing either production limit.
    complete_support = []
    for passage in [*descriptions.values(), example]:
        start = body.index(passage)
        source = next(item for item in originals
                      if item.selection.start <= start and start + len(passage) <= item.selection.end)
        complete_support.append(build_evidence(
            snapshot=source.snapshot, element=source.element, start=start, end=start + len(passage),
        ))
    assert all(len(item.excerpt) <= 1800 for item in complete_support)
    assert sum(len(item.excerpt) for item in complete_support) <= 6000
    assert requirement_coverage(task, complete_support, {
        item.id: [task.requirement_id] for item in complete_support
    })["missing_requirements"] == []

    packet, requirement_ids = select_evidence_packet(
        originals, budget=RetrievedEvidenceBudget(1800, 6000, 24), query=task.query,
        requirements_by_evidence={item.id: [task] for item in originals},
    )
    coverage = requirement_coverage(task, packet, requirement_ids)

    assert len(packet) <= 24
    assert all(len(item.excerpt) <= 1800 for item in packet)
    assert sum(len(item.excerpt) for item in packet) <= 6000
    for selected in packet:
        assert any(contains_evidence_range(original, selected) for original in originals)
        assert selected.excerpt == body[selected.selection.start:selected.selection.end]
    assert coverage["missing_aspects"] == []
    assert coverage["missing_needs"] == []
    assert coverage["missing_requirements"] == []


def _prepare_many_parameter_ranges():
    """Replay a many-parameter reference page without depending on live content."""
    from src.core.documents import DocumentElement, build_snapshot
    from src.core.evidence import build_evidence
    from src.runtime.nodes.synthesis.evidence_selection import requirement_coverage, select_evidence_packet

    aspects = ["sep", "dtype", "parse_dates", "na_values", "usecols", "converters", "chunksize", "iterator"]
    task = RetrievalTask(
        requirement_id="csv_parameters", route="docs", query="pandas.read_csv parameter explanations and examples",
        k=8, requirement={"library": "pandas", "symbols": ["pandas.read_csv"], "aspects": aspects,
                          "evidence_needs": ["parameters", "explanation", "examples"]},
    )
    fragments = [
        f"Parameters\n{aspect}: Controls the {aspect} setting when reading values from the input file.\n\n"
        + "Additional documentation about reading and converting input values.\n\n" * 40
        + "Examples\n>>> pandas.read_csv('input.csv')\n\n"
        for aspect in aspects
    ]
    body = "# pandas.read_csv\npandas.read_csv(filepath_or_buffer)\n\n" + "".join(fragments)
    snapshot = build_snapshot(
        source_uri="https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html", title="pandas.read_csv",
        media_type="text/markdown", source_type="official", content=body, parser="test", parser_version="1",
    )
    element = DocumentElement(element_id="body", kind="paragraph", text=body)
    evidence = []
    start = body.index(fragments[0])
    for fragment in fragments:
        evidence.append(build_evidence(snapshot=snapshot, element=element, start=start, end=start + len(fragment)))
        start += len(fragment)
    for per_item, total in ((1800, 6000), (900, 3000)):
        packet, ids = select_evidence_packet(
            evidence, budget=RetrievedEvidenceBudget(per_item, total, 24), query=task.query,
            requirements_by_evidence={item.id: [task] for item in evidence},
        )
        assert packet
        assert all(item.excerpt == body[item.selection.start:item.selection.end] for item in packet)
        assert sum(len(item.excerpt) for item in packet) <= total
        assert all(len(item.excerpt) <= per_item for item in packet)
        coverage = requirement_coverage(task, packet, ids)
        assert coverage["missing_aspects"] == []
        assert coverage["missing_needs"] == []
    return "normal and compact packets preserve all parameter and example coverage"


def test_many_parameter_reference_preparation_finishes_with_original_coverage():
    # A process timeout makes the regression fail without hanging the test run.
    # The allowance includes imports and is deliberately much larger than the
    # normal preparation time. All selection and coverage collaborators are real.
    source = "import runpy; print(runpy.run_path(" + repr(str(Path(__file__).resolve())) + ")[\"_prepare_many_parameter_ranges\"]())"
    result = subprocess.run(
        [sys.executable, "-c", source], cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, timeout=60, check=True,
    )
    assert "preserve all parameter and example coverage" in result.stdout


def _budget_case_source(text, key, *, upload=False):
    """Build immutable original ranges for the packet allocation cases below."""
    from src.core.documents import DocumentElement, build_snapshot
    from src.core.evidence import build_evidence

    snapshot = build_snapshot(
        source_uri=(f"upload://session/{key}" if upload else
                    f"https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#{key}"),
        title=f"{key}.py" if upload else "pandas.read_csv",
        media_type="text/x-python" if upload else "text/markdown",
        source_type="upload" if upload else "official",
        content=text, parser="test", parser_version="1",
    )
    return build_evidence(snapshot=snapshot, element=DocumentElement(
        element_id="body", kind="code" if upload else "paragraph", text=text,
        language="python" if upload else None, metadata={"file_id": key} if upload else {},
    ))


def _assert_budget_case_ranges(packet, originals, budget):
    from src.runtime.nodes.synthesis.evidence_selection import contains_evidence_range

    assert len(packet) <= budget.max_items
    assert all(len(item.excerpt) <= budget.max_excerpt_chars for item in packet)
    assert sum(len(item.excerpt) for item in packet) <= budget.max_total_excerpt_chars
    for item in packet:
        assert any(contains_evidence_range(original, item) for original in originals)
        assert item.excerpt == item.element.text[item.selection.start:item.selection.end]


def test_duplicate_noncontributing_docs_do_not_exhaust_items_before_required_parameter():
    """Repeated irrelevant parameter docs cannot consume the missing parameter's slot."""
    from src.runtime.nodes.synthesis.evidence_selection import requirement_coverage, select_evidence_packet

    task = RetrievalTask(
        requirement_id="csv_usecols", route="docs", query="pandas.read_csv usecols explanation", k=6,
        requirement={"library": "pandas", "symbols": ["pandas.read_csv"], "aspects": ["usecols"],
                     "evidence_needs": ["parameters", "explanation"]},
    )
    irrelevant = (
        "# pandas.read_csv\npandas.read_csv(filepath_or_buffer, encoding=None)\n\nParameters\n"
        "encoding: Selects the character encoding used when decoding input bytes.\n"
    )
    required = (
        "# pandas.read_csv\npandas.read_csv(filepath_or_buffer, usecols=None)\n\nParameters\n"
        "usecols: Selects the input columns to read by their names or integer positions.\n"
    )
    originals = [*[_budget_case_source(irrelevant, f"duplicate-{index}") for index in range(5)],
                 _budget_case_source(required, "required-usecols")]
    budget = RetrievedEvidenceBudget(350, 700, 2)
    assert requirement_coverage(task, originals, {
        item.id: [task.requirement_id] for item in originals
    })["missing_requirements"] == []

    packet, ids = select_evidence_packet(
        originals, budget=budget, query=task.query,
        requirements_by_evidence={item.id: [task] for item in originals},
    )

    _assert_budget_case_ranges(packet, originals, budget)
    assert any(item.snapshot == originals[-1].snapshot for item in packet)
    assert requirement_coverage(task, packet, ids)["missing_requirements"] == []


def test_two_docs_requirements_and_upload_share_space_for_their_required_ranges():
    """Two long docs cannot displace a required uploaded file when all support fits."""
    from src.runtime.nodes.synthesis.evidence_selection import requirement_coverage, select_evidence_packet

    docs_tasks = [RetrievalTask(
        requirement_id=f"csv_{aspect}", route="docs", query=f"pandas.read_csv {aspect} explanation", k=1,
        requirement={"library": "pandas", "symbols": ["pandas.read_csv"], "aspects": [aspect],
                     "evidence_needs": ["parameters", "explanation"]},
    ) for aspect in ("sep", "usecols")]
    upload_task = RetrievalTask(
        requirement_id="uploaded_batch_size", route="upload", query="batch_size setting", k=1,
        requirement={"file_ids": ["job"], "aspects": ["batch_size"], "match": "topic"},
    )
    background = "Background about loading input files and processing records. " * 15
    originals = [_budget_case_source(
        "# pandas.read_csv\npandas.read_csv(filepath_or_buffer)\n\nParameters\n"
        f"{aspect}: {description}\n\nNotes\n{background}\n", f"fair-{aspect}",
    ) for aspect, description in (
        ("sep", "Sets the delimiter separating input columns."),
        ("usecols", "Selects the input columns by names or integer positions."),
    )]
    originals.append(_budget_case_source("batch_size = 1000\n\nprocess_records(batch_size)\n", "job", upload=True))
    tasks = [*docs_tasks, upload_task]
    associations = {item.id: [task] for item, task in zip(originals, tasks)}
    original_ids = {item.id: [task.requirement_id] for item, task in zip(originals, tasks)}
    for task in tasks:
        assert requirement_coverage(task, originals, original_ids)["is_partial"] is False
    budget = RetrievedEvidenceBudget(350, 600, 3)

    packet, ids = select_evidence_packet(
        originals, budget=budget, query="Compare sep, usecols and uploaded batch_size",
        requirements_by_evidence=associations,
    )

    _assert_budget_case_ranges(packet, originals, budget)
    assert {requirement_id for associated in ids.values() for requirement_id in associated} == {
        task.requirement_id for task in tasks
    }
    for task in tasks:
        coverage = requirement_coverage(task, packet, ids)
        assert coverage["evidence_ids"]
        assert coverage["missing_aspects"] == []
        assert coverage["is_partial"] is False
    assert requirement_coverage(upload_task, packet, ids)["missing_file_ids"] == []


def test_impossible_description_budget_preserves_limits_and_reports_missing_coverage():
    """A clipped description stays explicitly incomplete when its full line cannot fit."""
    from src.runtime.nodes.synthesis.evidence_selection import (
        missing_evidence_requirement_ids, requirement_coverage, select_evidence_packet,
    )

    task = RetrievalTask(
        requirement_id="csv_long_usecols", route="docs", query="pandas.read_csv usecols explanation", k=1,
        requirement={"library": "pandas", "symbols": ["pandas.read_csv"], "aspects": ["usecols"],
                     "evidence_needs": ["parameters", "explanation"]},
    )
    description = (
        "usecols: Selects columns by their names or integer positions. "
        + "The selected columns keep their input ordering unless explicitly reordered afterward. " * 8
        + "\n"
    )
    original = _budget_case_source(
        "# pandas.read_csv\npandas.read_csv(filepath_or_buffer, usecols=None)\n\nParameters\n"
        + description, "unavoidably-long-usecols",
    )
    budget = RetrievedEvidenceBudget(120, 180, 2)
    assert len(description) > budget.max_total_excerpt_chars
    assert requirement_coverage(task, [original], {original.id: [task.requirement_id]})["is_partial"] is False

    packet, ids = select_evidence_packet(
        [original], budget=budget, query=task.query,
        requirements_by_evidence={original.id: [task]},
    )

    _assert_budget_case_ranges(packet, [original], budget)
    coverage = requirement_coverage(task, packet, ids)
    assert coverage["is_partial"] is True
    assert coverage["missing_aspects"] == ["usecols"]
    assert "aspect:usecols" in coverage["missing_requirements"]
    assert missing_evidence_requirement_ids(PlannerOutput(use_retrieval=True, tasks=[task]), packet, ids) == [
        task.requirement_id
    ]


def test_shared_doc_kinds_do_not_consume_slots_needed_by_two_parameter_requirements():
    """Cheaper generic prose cannot use the only slots for two complete requirements."""
    from src.runtime.nodes.synthesis.evidence_selection import requirement_coverage, select_evidence_packet

    tasks = [RetrievalTask(
        requirement_id=f"csv_{aspect}", route="docs", query=f"pandas.read_csv {aspect} explanation", k=2,
        requirement={"library": "pandas", "symbols": ["pandas.read_csv"], "aspects": [aspect],
                     "evidence_needs": ["parameters", "explanation"]},
    ) for aspect in ("sep", "usecols")]
    header = "# pandas.read_csv\npandas.read_csv(filepath_or_buffer)\n\nParameters\n"
    originals = []
    required_sources = []
    associations = {}
    for task, aspect in zip(tasks, ("sep", "usecols")):
        generic = _budget_case_source(
            header + "encoding: Uses the specified text encoding.\n", f"generic-{aspect}",
        )
        required = _budget_case_source(
            header + f"{aspect}: Selects the input values according to the supplied parameter and preserves "
            "the expected output behavior for each row in the file.\n", f"required-{aspect}",
        )
        originals.extend([generic, required])
        required_sources.append(required)
        associations.update({item.id: [task] for item in (generic, required)})
    budget = RetrievedEvidenceBudget(250, 500, 2)
    # Selecting each complete requirement's source fits all three ceilings.
    _assert_budget_case_ranges(required_sources, originals, budget)
    required_ids = {item.id: [task.requirement_id] for item, task in zip(required_sources, tasks)}
    for task in tasks:
        assert requirement_coverage(task, required_sources, required_ids)["is_partial"] is False

    packet, ids = select_evidence_packet(
        originals, budget=budget, query="Explain sep and usecols",
        requirements_by_evidence=associations,
    )

    _assert_budget_case_ranges(packet, originals, budget)
    for task in tasks:
        assert requirement_coverage(task, packet, ids)["missing_requirements"] == []
