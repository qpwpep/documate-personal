"""Documentation candidates remain bounded without interpreting source meaning."""

import pytest
import requests

from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import RetrievalRequirement
from src.infra.settings import AppSettings
from src.infra.tools.docs_search.ranking import select_docs_candidates
from src.infra.tools.docs_search.tool import build_docs_search_tool


def source_hit(body, *, symbol="numpy.mean", start=0, end=None, version="stable", score=0.4):
    snapshot = build_snapshot(source_uri=f"https://numpy.org/doc/{version}/reference/generated/{symbol}.html",
                              title=symbol, media_type="text/markdown", source_type="official", content=body,
                              parser="test", parser_version="1")
    evidence = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=body),
                              start=start, end=min(len(body), 3200) if end is None else end)
    return SearchHit(evidence=evidence, rank=1, score=RetrievalScore(metric="provider", direction="higher", normalized=score))


def select(hits, *, k=1, requirement=None, preserved_evidence_ids=None):
    return select_docs_candidates("numpy.mean axis", hits, requirement or RetrievalRequirement(
        library="numpy", symbols=["numpy.mean"], aspects=["axis"], evidence_needs=["parameters", "explanation"],
    ), k=k, allowed_domains=["numpy.org"], preserved_evidence_ids=preserved_evidence_ids)


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
