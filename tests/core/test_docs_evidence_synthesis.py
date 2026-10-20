"""Bounded documentation ranges survive allocation using explicit semantic verdicts."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import requests
import pytest

from src.core.evidence import SearchHit
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.tool import build_docs_search_tool
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
from src.runtime.nodes.synthesis.evidence_selection import (
    missing_evidence_requirement_ids,
    prepare_evidence_candidates,
    requirement_coverage,
    select_evidence_packet,
)


def _retrieve(monkeypatch, body, *, need, symbol="numpy.mean", aspects=None, library="numpy"):
    """Only HTTP is fake; search and bounded candidate preparation stay real."""
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
    assert result.diagnostics.answerability == "unknown"
    assert result.hits
    hits = [SearchHit.model_validate(hit).model_copy(update={"requirement_id": task.requirement_id})
            for hit in result.hits]
    return task, hits


def _hits(originals, tasks):
    return [SearchHit(evidence=item, requirement_id=task.requirement_id, rank=index + 1,
                      score={"metric": "test", "direction": "higher", "raw": 1.0})
            for index, (item, task) in enumerate(zip(originals, tasks))]


def _fixture_support(candidates, task, verdicts):
    """Attach explicit verdict fixtures to real candidates containing exact fixture passages.

    This only locates ranges. Each test supplies every semantic label; no text
    pattern or requirement is converted into a semantic verdict here.
    """
    support = {}
    for passage, labels in verdicts:
        matching = [hit.evidence for hit in candidates
                    if hit.requirement_id == task.requirement_id and passage in hit.evidence.excerpt]
        assert matching, f"No bounded candidate preserves fixture passage: {passage!r}"
        candidate = min(matching, key=lambda item: (len(item.excerpt), item.selection.start))
        support.setdefault(candidate.id, {}).setdefault(task.requirement_id, set()).update(labels)
    return support


def _prepare(task, hits, *, verdicts, excerpt_chars=1800, total_chars=6000, max_items=8):
    planner = PlannerOutput(use_retrieval=True, tasks=[task])
    budget = RetrievedEvidenceBudget(excerpt_chars, total_chars, max_items)
    candidates = prepare_evidence_candidates(
        user_input=task.query, hits=hits, planner_output=planner, budget=budget,
    )
    packet, ids = select_evidence_packet(
        [hit.evidence for hit in candidates], budget=budget, query=task.query,
        requirements_by_evidence={hit.evidence.id: [task] for hit in candidates},
        semantic_support=_fixture_support(candidates, task, verdicts),
    )
    assert {item.id for item in packet} <= {hit.evidence.id for hit in candidates}
    prepared = SimpleNamespace(
        evidence_packet=packet,
        missing_requirement_ids=missing_evidence_requirement_ids(planner, packet, ids),
    )
    return prepared, requirement_coverage(task, packet, ids)


def test_multiline_signature_survives_budget_beside_a_long_unbroken_description(monkeypatch):
    excerpt_chars = 120
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

    prepared, coverage = _prepare(task, hits, excerpt_chars=excerpt_chars, verdicts=[
        (signature, {"relevance", "need:signature"}),
        (description, {"relevance", "need:explanation"}),
    ])

    assert coverage["is_partial"] is False
    assert prepared.missing_requirement_ids == []
    assert any(signature in item.excerpt for item in prepared.evidence_packet)
    assert any(description in item.excerpt for item in prepared.evidence_packet)
    _assert_budget_case_ranges(
        prepared.evidence_packet, [hit.evidence for hit in hits], RetrievedEvidenceBudget(excerpt_chars, 6000, 8),
    )


@pytest.mark.parametrize(("clipped_start", "clipped_end", "excerpt_chars"), [(1, 0, 1800), (0, 1, 1800), (0, 0, 40)])
def test_signature_outside_retrieved_range_or_budget_is_not_reconstructed(clipped_start, clipped_end, excerpt_chars):
    from src.core.evidence import build_evidence

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

    candidates = prepare_evidence_candidates(
        user_input=task.query, hits=_hits([original], [task]),
        planner_output=PlannerOutput(use_retrieval=True, tasks=[task]), budget=budget,
    )
    assert candidates
    for hit in candidates:
        _assert_budget_case_ranges([hit.evidence], [original], budget)
    assert all(signature not in hit.evidence.excerpt for hit in candidates)
    # The semantic stage can reject these incomplete candidates. The allocator
    # consumes that rejection instead of trying to infer a signature itself.
    packet, ids = select_evidence_packet(
        [hit.evidence for hit in candidates], budget=budget, query=task.query,
        requirements_by_evidence={hit.evidence.id: [task] for hit in candidates}, semantic_support={},
    )
    assert packet == []
    assert requirement_coverage(task, packet, ids) == {"evidence_ids": [], "is_partial": True}


@pytest.mark.parametrize(("symbol", "excerpt_chars"), [("numpy.mean", 1800), ("np.mean", 200)])
def test_definition_context_preserves_alias_candidates_before_semantic_selection(monkeypatch, symbol, excerpt_chars):
    introduction = "Compute the arithmetic mean along the specified axis."
    condition = "If axis is None, all dimensions are reduced to a scalar result."
    body = (
        "# numpy.mean\nnumpy.mean(a, axis=None)\n\n" + introduction + "\n\n"
        + "The function describes general arithmetic operations on input arrays. " * 28
        + "\n\n" + condition + "\n"
    )
    task, hits = _retrieve(monkeypatch, body, need=["parameters", "explanation"],
                           symbol=symbol, aspects=["axis"])

    prepared, coverage = _prepare(task, hits, excerpt_chars=excerpt_chars, verdicts=[
        (introduction, {"relevance", "need:explanation"}),
        (condition, {"relevance", "aspect:axis", "need:parameters"}),
    ])

    assert coverage["is_partial"] is False
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
    from src.runtime.nodes.synthesis.evidence_selection import contains_evidence_range

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
    assert requirement_coverage(task, originals, original_ids)["is_partial"] is False

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
    })["is_partial"] is False

    prepared, coverage = _prepare(task, _hits(originals, [task] * len(originals)), max_items=24,
        verdicts=[(description.strip(), {"relevance", f"aspect:{name}", "need:parameters", "need:explanation"})
                  for name, description in descriptions.items()]
                 + [(example.strip(), {"relevance", "need:examples"})])
    packet = prepared.evidence_packet

    assert len(packet) <= 24
    assert all(len(item.excerpt) <= 1800 for item in packet)
    assert sum(len(item.excerpt) for item in packet) <= 6000
    for selected in packet:
        assert any(contains_evidence_range(original, selected) for original in originals)
        assert selected.excerpt == body[selected.selection.start:selected.selection.end]
    assert coverage["is_partial"] is False
    assert all(any(description.strip() in item.excerpt for item in packet) for description in descriptions.values())
    assert any(example.strip() in item.excerpt for item in packet)


def _prepare_many_parameter_ranges():
    """Replay a many-parameter reference page without depending on live content."""
    from src.core.documents import DocumentElement, build_snapshot
    from src.core.evidence import build_evidence

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
        prepared, coverage = _prepare(task, _hits(evidence, [task] * len(evidence)),
            excerpt_chars=per_item, total_chars=total, max_items=24,
            verdicts=[
                (f"{aspect}: Controls the {aspect} setting when reading values from the input file.",
                 {"relevance", f"aspect:{aspect}", "need:parameters", "need:explanation"})
                for aspect in aspects
            ] + [("Examples\n>>> pandas.read_csv('input.csv')", {"relevance", "need:examples"})])
        packet = prepared.evidence_packet
        assert packet
        assert all(item.excerpt == body[item.selection.start:item.selection.end] for item in packet)
        assert sum(len(item.excerpt) for item in packet) <= total
        assert all(len(item.excerpt) <= per_item for item in packet)
        assert coverage["is_partial"] is False
        assert all(any(f"{aspect}: Controls the {aspect} setting" in item.excerpt for item in packet)
                   for aspect in aspects)
        assert any(">>> pandas.read_csv('input.csv')" in item.excerpt for item in packet)
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


def test_two_docs_requirements_and_upload_share_space_for_their_required_ranges():
    """Two long docs cannot displace a required uploaded file when all support fits."""

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
    original_ids = {item.id: [task.requirement_id] for item, task in zip(originals, tasks)}
    for task in tasks:
        assert requirement_coverage(task, originals, original_ids)["is_partial"] is False
    budget = RetrievedEvidenceBudget(350, 600, 3)
    candidates = prepare_evidence_candidates(
        user_input="Compare sep, usecols and uploaded batch_size", hits=_hits(originals, tasks),
        planner_output=PlannerOutput(use_retrieval=True, tasks=tasks), budget=budget,
    )
    support = {}
    for task, aspect, description in zip(docs_tasks, ("sep", "usecols"), (
        "Sets the delimiter separating input columns.", "Selects the input columns by names or integer positions.",
    )):
        support.update(_fixture_support(candidates, task, [(f"{aspect}: {description}", {
            "relevance", f"aspect:{aspect}", "need:parameters", "need:explanation",
        })]))
    tasks_by_id = {task.requirement_id: task for task in tasks}

    packet, ids = select_evidence_packet(
        [hit.evidence for hit in candidates], budget=budget, query="Compare sep, usecols and uploaded batch_size",
        requirements_by_evidence={hit.evidence.id: [tasks_by_id[hit.requirement_id]] for hit in candidates},
        semantic_support=support,
    )

    _assert_budget_case_ranges(packet, originals, budget)
    assert {requirement_id for associated in ids.values() for requirement_id in associated} == {
        task.requirement_id for task in tasks
    }
    for task in tasks:
        coverage = requirement_coverage(task, packet, ids)
        assert coverage["evidence_ids"]
        assert coverage["is_partial"] is False
    assert requirement_coverage(upload_task, packet, ids)["missing_file_ids"] == []
    assert requirement_coverage(upload_task, packet, ids)["missing_aspects"] == []


def test_impossible_description_budget_preserves_ranges_and_consumes_semantic_rejection():
    """Clipped candidates have new identities and cannot borrow the full source's verdict."""

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

    candidates = prepare_evidence_candidates(
        user_input=task.query, hits=_hits([original], [task]),
        planner_output=PlannerOutput(use_retrieval=True, tasks=[task]), budget=budget,
    )
    assert candidates
    for hit in candidates:
        _assert_budget_case_ranges([hit.evidence], [original], budget)
        assert hit.evidence.id != original.id
        assert description not in hit.evidence.excerpt
    # A previous full-source decision is not a verdict for any bounded candidate.
    packet, ids = select_evidence_packet(
        [hit.evidence for hit in candidates], budget=budget, query=task.query,
        requirements_by_evidence={hit.evidence.id: [task] for hit in candidates},
        semantic_support={original.id: {task.requirement_id: {
            "relevance", "aspect:usecols", "need:parameters", "need:explanation",
        }}},
    )
    _assert_budget_case_ranges(packet, [original], budget)
    assert packet == []
    coverage = requirement_coverage(task, packet, ids)
    assert coverage["is_partial"] is True
    assert missing_evidence_requirement_ids(PlannerOutput(use_retrieval=True, tasks=[task]), packet, ids) == [
        task.requirement_id
    ]
