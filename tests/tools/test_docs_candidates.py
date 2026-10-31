"""Documentation candidates remain bounded without interpreting source meaning."""

import json

import pytest
import requests

pytestmark = pytest.mark.usefixtures("unavailable_official_pages")

from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalRequirement, RetrievalTask
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.policy import is_current_stable_doc_url
from src.infra.tools.docs_search.ranking import select_docs_candidates
from src.infra.tools.docs_search.tool import build_docs_search_tool
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget
from src.runtime.nodes.synthesis.evidence_selection import prepare_evidence_candidates


def source_hit(body, *, symbol="numpy.mean", start=0, end=None, version="stable", score=0.4, title=None, url_suffix=""):
    snapshot = build_snapshot(source_uri=f"https://numpy.org/doc/{version}/reference/generated/{symbol}.html{url_suffix}",
                              title=title or symbol, media_type="text/markdown", source_type="official", content=body,
                              parser="test", parser_version="1")
    evidence = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=body),
                              start=start, end=min(len(body), 3200) if end is None else end)
    return SearchHit(evidence=evidence, rank=1, score=RetrievalScore(metric="provider", direction="higher", normalized=score))


def select(hits, *, k=1, requirement=None, preserved_evidence_ids=None):
    return select_docs_candidates("numpy.mean axis", hits, requirement or RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], aspects=["axis"], evidence_needs=["parameters", "explanation"],
    ), k=k, allowed_domains=["numpy.org"], preserved_evidence_ids=preserved_evidence_ids)


@pytest.mark.parametrize("url,expected", [
    ("https://pandas.pydata.org/docs/reference/api/pandas.merge.html", True),
    ("https://pandas.pydata.org/docs/dev/reference/api/pandas.merge.html", False),
    ("https://pandas.pydata.org/pandas-docs/version/2.2.2/reference/api/pandas.merge.html", False),
    ("https://docs.python.org/ko/3/library/pathlib.html", True),
    ("https://docs.python.org/3.12/library/pathlib.html", False),
    ("https://numpy.org/doc/stable/reference/generated/numpy.mean.html", True),
    ("https://numpy.org/doc/dev/reference/generated/numpy.mean.html", False),
    ("https://docs.pytorch.org/docs/main/generated/torch.mean.html", False),
    ("https://docs.pydantic.dev/latest/api/base_model/", True),
    ("https://huggingface.co/docs/transformers/main/en/index", False),
    ("https://pandas.pydata.org.attacker.example/docs/reference/api/pandas.merge.html", False),
    ("https://pandas.pydata.org/docs/%2e%2e/community/", False),
])
def test_current_stable_alias_is_distinct_from_official_url_permission(url, expected):
    assert is_current_stable_doc_url(url) is expected


@pytest.mark.parametrize("k,limit", [(1, 6), (3, 9), (20, 20)])
def test_candidate_count_window_size_and_source_balance(k, limit):
    first = source_hit("".join(f"# numpy.operation_{i}\nnumpy.operation_{i}(a, axis=None)\naxis: details.\n" for i in range(150)))
    second = source_hit("The mean is computed across the chosen axis.\n", symbol="numpy.other", score=0.01)
    candidates = select([first, second], k=k)
    assert 2 <= len(candidates) <= limit
    assert {first.evidence.snapshot.snapshot_id, second.evidence.snapshot.snapshot_id} <= {
        hit.evidence.snapshot.snapshot_id for hit in candidates[:2]
    }
    assert all(0 < len(hit.evidence.excerpt) <= 3200 for hit in candidates)
    assert len({hit.evidence.id for hit in candidates}) == len(candidates)


def test_unspecified_version_keeps_stable_windows_before_archived_or_development_sources():
    body = (
        "# numpy.mean\nnumpy.mean(a, axis=None)\n"
        + "The axis parameter controls the selected dimension.\n" * 100
        + "\n## Examples\n>>> np.mean([[1, 2], [3, 4]])\n2.5\n"
    )
    stable = source_hit(body, score=0.01)
    alternatives = [source_hit("numpy.mean(a, axis=None)\nOlder or development description.",
                               version=version, score=0.99)
                    for version in ["1.20", "1.21", "1.22", "1.23", "1.24", "dev"]]

    candidates = select([*alternatives, stable])

    assert candidates[0].evidence.snapshot.source_uri == stable.evidence.snapshot.source_uri
    assert any(">>> np.mean([[1, 2], [3, 4]])" in hit.evidence.excerpt for hit in candidates)
    stable_positions = [index for index, hit in enumerate(candidates)
                        if hit.evidence.snapshot.snapshot_id == stable.evidence.snapshot.snapshot_id]
    assert stable_positions == list(range(len(stable_positions)))
    assert len(candidates) <= 6
    for hit in candidates:
        evidence = hit.evidence
        assert evidence.excerpt == evidence.element.text[evidence.selection.start:evidence.selection.end]


def test_new_development_snapshot_does_not_displace_current_stable_source():
    stable = source_hit("numpy.mean(a)\nCurrent stable explanation.", score=0.01)
    development = source_hit("numpy.mean(a)\nDevelopment explanation.", version="dev", score=0.99)
    candidates = select_docs_candidates("numpy.mean", [development, stable],
        RetrievalRequirement(library="numpy", symbols=["numpy.mean"]), k=1,
        allowed_domains=["numpy.org"], preferred_snapshot_ids={development.evidence.snapshot.snapshot_id})

    assert [hit.evidence.id for hit in candidates] == [stable.evidence.id, development.evidence.id]


def test_explicit_version_keeps_existing_candidate_order_and_original_urls():
    requested = source_hit("numpy.mean(a)\nVersioned explanation.", version="1.26", score=0.99)
    stable = source_hit("numpy.mean(a)\nCurrent stable explanation.", score=0.01)
    candidates = select([stable, requested], requirement=RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], version="1.26"))

    assert [hit.evidence.id for hit in candidates] == [requested.evidence.id, stable.evidence.id]
    assert candidates[0].evidence.snapshot.source_uri == requested.evidence.snapshot.source_uri


@pytest.mark.parametrize("version,expected_url,expected_limit", [
    ("1.5.3", "https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.merge.html", 9),
    (None, "https://pandas.pydata.org/docs/reference/api/pandas.merge.html", 6),
])
def test_small_search_budget_recovers_explicit_version_without_expanding_default_search(
    monkeypatch, version, expected_url, expected_limit,
):
    requested = {
        "url": "https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.merge.html",
        "title": "pandas.merge - pandas 1.5.3 documentation",
        "raw_content": "pandas.merge(left, right, how='inner', copy=True)\nMerge DataFrames using matching keys.",
        "score": 0.84,
    }
    current = {
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.merge.html",
        "title": "pandas.merge - pandas 3.1.0 documentation",
        "raw_content": "pandas.merge(left, right, how='inner')\nMerge DataFrames using matching keys.",
        "score": 0.46,
    }
    ordered = {
        "url": "https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.merge_ordered.html",
        "title": "pandas.merge_ordered - pandas 1.5.3 documentation",
        "raw_content": "pandas.merge_ordered(left, right)\nMerge ordered data with optional interpolation.",
        "score": 0.8,
    }
    searches = []

    def post(url, **kwargs):
        searches.append(kwargs["json"])
        # The observed six-result response omitted the requested API/version;
        # the nine-result response included it ahead of alternative APIs.
        results = [ordered, current]
        if kwargs["json"]["max_results"] >= 9:
            results.insert(0, requested)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"results": results}).encode()
        return response

    monkeypatch.setattr(requests, "post", post)
    tool = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test"))
    requirement = RetrievalRequirement(library="pandas", symbols=["pandas.merge"], version=version,
                                       evidence_needs=["explanation"])
    first = tool("pandas merge usage", requirement=requirement, k=1)
    matching = [hit for hit in first.hits if hit.evidence.snapshot.source_uri == expected_url]

    assert matching
    assert matching[0].evidence.snapshot.title == (requested if version else current)["title"]
    assert matching[0].evidence.element.text == (requested if version else current)["raw_content"]
    assert len(first.hits) <= 6
    assert first.diagnostics.completed_requests[0].max_results == expected_limit
    second = tool("pandas merge usage", requirement=requirement, k=2, previous_hits=first.hits,
                  completed_requests=first.diagnostics.completed_requests)
    assert {hit.evidence.id for hit in second.hits} == {hit.evidence.id for hit in first.hits}
    assert [search["max_results"] for search in searches] == [expected_limit]


def test_explicit_version_keeps_primary_api_middle_and_examples_available_for_validation():
    parameter = "axis: Select which dimension to reduce; None reduces every dimension."
    example = ">>> numpy.mean([[1, 2], [3, 4]], axis=0)"
    body = (
        "Navigation links.\n" * 130
        + "\nnumpy.mean(a, axis=None)\nCompute the arithmetic mean.\n"
        + "Basic operation details.\n" * 120
        + "\n## Parameters\n" + parameter + "\n"
        + "Parameter context and supporting discussion.\n" * 80
        + "\n## Examples\n" + example + "\n[2., 3.]\n"
    )
    requested = source_hit(body, version="1.5", title="numpy.mean - NumPy 1.5.3 documentation", score=0.1)
    related = [source_hit(body.replace("numpy.mean", symbol), symbol=symbol, version="1.5",
                          title=f"{symbol} - NumPy 1.5.3 documentation", score=0.9)
               for symbol in ["numpy.mean_asof", "numpy.mean_ordered"]]
    others = [source_hit(body, version=version, score=0.99) for version in ["stable", "2.1", "3.1"]]
    task = RetrievalTask(route="docs", query="numpy.mean 1.5.3 usage", k=1,
                         requirement=RetrievalRequirement(library="numpy", symbols=["numpy.mean"],
                                                          version="1.5.3", evidence_needs=["explanation"]).model_dump())
    candidates = select_docs_candidates(task.query, [*related, *others, requested], task.requirement,
                                       k=1, allowed_domains=["numpy.org"])
    prepared = prepare_evidence_candidates(
        user_input=task.query,
        hits=[hit.model_copy(update={"requirement_id": task.requirement_id}) for hit in candidates],
        planner_output=PlannerOutput(use_retrieval=True, tasks=[task]),
        budget=RetrievedEvidenceBudget(1800, 6000, 8),
    )
    selected_text = "\n".join(hit.evidence.excerpt for hit in prepared
                              if hit.evidence.snapshot.snapshot_id == requested.evidence.snapshot.snapshot_id)

    assert parameter in selected_text
    assert example in selected_text
    assert len(candidates) <= 6
    assert candidates[0].evidence.snapshot.source_uri == requested.evidence.snapshot.source_uri
    assert all(hit.evidence.excerpt == hit.evidence.element.text[hit.evidence.selection.start:hit.evidence.selection.end]
               for hit in candidates)


@pytest.mark.parametrize("other_version", ["1.5.30", "1.5.3rc1", "1.5.3-rc1"])
def test_explicit_version_priority_uses_complete_tokens_and_preserves_other_candidates(other_version):
    requested = source_hit("numpy.mean(a)\nRequested explanation.", version="1.5",
                           title="numpy.mean - NumPy 1.5.3 documentation", score=0.01)
    other = source_hit("numpy.mean(a)\nOther explanation.", version="1.5.30",
                       title=f"numpy.mean - NumPy {other_version} documentation", score=0.99)
    query_only = source_hit("numpy.mean(a)\nCurrent explanation.", score=0.99,
                            url_suffix="?version=1.5.3#1.5.3")
    candidates = select([other, query_only, requested], requirement=RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], version="1.5.3"))

    assert candidates[0].evidence.id == requested.evidence.id
    assert {hit.evidence.id for hit in candidates} == {requested.evidence.id, other.evidence.id, query_only.evidence.id}


def test_unparsed_source_keeps_base_window_and_late_literal_anchor():
    body = "이 문서는 일반 문장으로만 작성되어 있습니다.\n" * 250 + "axis를 None으로 설정하면 전체 원소를 사용합니다.\n"
    original = source_hit(body)
    candidates = select([original])
    assert original.evidence.id in {hit.evidence.id for hit in candidates}
    assert any("axis를 None" in hit.evidence.excerpt for hit in candidates)


def test_exact_declaration_window_beats_incidental_cross_reference_with_same_terms():
    requested_behavior = "axis=None reduces every dimension; axis=0 reduces only the first dimension."
    body = (
        "# Array overview\n"
        "The numpy.mean function accepts axis; see its reference for the full contract.\n"
        + "Unrelated introductory context.\n" * 200
        + "\n# numpy.other\nnumpy.other(a)\nOther API example and details.\n" * 10
        + "\n# numpy.mean\nnumpy.mean(a, axis=None)\n" + requested_behavior + "\n"
    )
    sources = [source_hit(body, symbol=f"numpy.module_{index}") for index in range(6)]
    candidates = select(sources, k=1)
    assert len(candidates) == 6
    assert all(requested_behavior in hit.evidence.excerpt for hit in candidates)
    assert all(hit.evidence.selection.start >= body.index("# numpy.mean") for hit in candidates)


def test_declaration_window_spends_budget_after_definition_not_previous_api():
    behavior = "axis=None reduces every dimension of the supplied array."
    declaration = "numpy.mean(a, axis=None)\n"
    body = (
        "# numpy.other\nnumpy.other(a)\n" + "Other API example and explanation.\n" * 140
        + "\n" + declaration + "Definition context.\n" * 145 + behavior + "\n"
    )
    sources = [source_hit(body, symbol=f"numpy.module_{index}") for index in range(6)]
    candidates = select(sources, k=1)
    assert all(behavior in hit.evidence.excerpt for hit in candidates)
    assert all(hit.evidence.excerpt.startswith(declaration) for hit in candidates)


def test_qualified_stdlib_symbol_keeps_relative_declaration_and_return_contract():
    declaration = "Path.exists(*, follow_symlinks=True)\n"
    behavior = "Return True if the path points to an existing file or directory, and False otherwise."
    body = (
        "pathlib — Object-oriented filesystem paths\n\n"
        "Path.home()\nSee pathlib.Path.exists for existence checks.\n\n"
        ">>> Path.home()\nPosixPath('/home/user')\n\n"
        + "Other path operation details.\n" * 180
        + declaration + behavior + "\n\n"
        + "Path.is_file(*, follow_symlinks=True)\nReturn whether this is a regular file.\n" * 50
    )
    snapshot = build_snapshot(
        source_uri="https://docs.python.org/3/library/pathlib.html",
        title="pathlib — Object-oriented filesystem paths — Python documentation",
        media_type="text/html", source_type="official", content=body,
        parser="official_html", parser_version="1",
    )
    original = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=body))
    task = RetrievalTask(requirement_id="exists", route="docs", query="pathlib.Path.exists 반환값", k=1,
        requirement={"library": "Python", "symbols": ["pathlib.Path.exists"], "evidence_needs": ["explanation"]})

    hits = select_docs_candidates(task.query, [SearchHit(evidence=original, rank=1,
        requirement_id=task.requirement_id, score={"metric": "provider", "direction": "higher"})],
        task.requirement, k=1, allowed_domains=["docs.python.org"])
    candidates = prepare_evidence_candidates(user_input=task.query, hits=hits,
        planner_output=PlannerOutput(use_retrieval=True, tasks=[task]),
        budget=RetrievedEvidenceBudget(1800, 6000, 8))

    assert hits[0].evidence.excerpt.startswith(declaration)
    assert behavior in candidates[0].evidence.excerpt
    assert all(hit.evidence.excerpt == body[hit.evidence.selection.start:hit.evidence.selection.end]
               for hit in [*hits, *candidates])


def test_overlapping_anchor_windows_do_not_displace_late_examples():
    body = (
        "# numpy.mean\nnumpy.mean(a, axis=None)\n"
        + "numpy.mean computes a result; axis chooses the reduction dimension.\n" * 100
        + "\n## Examples\n>>> np.mean([[1, 2], [3, 4]])\n2.5\n"
    )
    candidates = select([source_hit(body)], requirement=RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], aspects=["axis"],
        evidence_needs=["parameters", "explanation", "examples"],
    ))
    assert any(">>> np.mean([[1, 2], [3, 4]])" in hit.evidence.excerpt for hit in candidates)
    covered = {offset for hit in candidates for offset in range(
        hit.evidence.selection.start, hit.evidence.selection.end,
    )}
    assert len(covered) == len(body)


@pytest.mark.parametrize("preserve", [False, True])
def test_tool_prioritizes_only_explicitly_preserved_previous_selection(monkeypatch, preserve):
    body = "Previous range without the target.\n" + "Introductory material.\n" * 250 + "numpy.mean(a, axis=None)\n"
    previous = source_hit(body, start=0, end=33)
    other_sources = [source_hit("numpy.mean(a, axis=None)\n", symbol=f"numpy.other_{index}") for index in range(5)]

    def post(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"results": []}'
        return response

    monkeypatch.setattr(requests, "post", post)
    tool = build_docs_search_tool(AppSettings(_env_file=None, tavily_api_key="test", openai_api_key="test"))
    result = tool("numpy.mean axis", k=1, requirement=RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], aspects=["axis"],
    ), previous_hits=[previous, *other_sources],
        preserved_evidence_ids=[previous.evidence.id] if preserve else [])

    assert len(result.hits) == 6
    assert (previous.evidence.id in {hit.evidence.id for hit in result.hits}) is preserve
    if not preserve:
        assert all("numpy.mean(a, axis=None)" in hit.evidence.excerpt for hit in result.hits)


def test_cropped_ranges_and_revised_sources_have_distinct_candidate_ids():
    body = "numpy.mean(a, axis=None)\nIf axis is None, all input dimensions are reduced.\n"
    full = source_hit(body)
    cropped = source_hit(body, start=body.index("If axis"), end=body.index("dimensions"))
    revised = source_hit(body.replace("None", "0"))
    candidates = select([full, cropped, revised], preserved_evidence_ids={cropped.evidence.id})
    assert {full.evidence.id, cropped.evidence.id, revised.evidence.id} <= {hit.evidence.id for hit in candidates}


@pytest.mark.parametrize("url", [
    "http://numpy.org/doc/stable/reference/generated/numpy.mean.html",
    "https://numpy.org.attacker.example/doc/stable/reference/generated/numpy.mean.html",
    "https://numpy.org/community/numpy.mean.html",
    "https://numpy.org/doc/../community/numpy.mean.html",
    "https://pandas.pydata.org/docs/reference/api/pandas.mean.html",
])
def test_previous_candidates_are_rechecked_against_authority(url):
    hit = source_hit("numpy.mean(a)\nCalculate the mean.")
    snapshot = build_snapshot(source_uri=url, title="numpy.mean", media_type="text/markdown", source_type="official",
                              content=hit.evidence.element.text, parser="test", parser_version="1")
    evidence = build_evidence(snapshot=snapshot, element=hit.evidence.element)
    assert select([hit.model_copy(update={"evidence": evidence})]) == []
