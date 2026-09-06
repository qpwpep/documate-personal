import pytest
from hypothesis import given, strategies as st

from src.core.answer_schema import finalize_answer, text_document
from src.core.documents import DocumentElement, SourceAnchor, TableCell, TableData
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.eval.config_models import BenchmarkCase
from src.eval.metric_rules import score_citation_traceability, score_reference_coverage
from src.runtime.nodes.synthesis.prompt_builder import prepare_evidence_packet
from tests.eval.response_fixtures import source_evidence


def test_budgeted_synthesis_packet_citation_traces_to_the_larger_search_selection() -> None:
    """A budgeted subrange stays traceable to the search result that supplied it."""
    source = source_evidence(text="앞부분. " + "검색으로 확인한 긴 문서 내용입니다. " * 20)
    observed = build_evidence(snapshot=source.snapshot, element=source.element, start=5, end=300)
    hit = SearchHit(evidence=observed, rank=1, score=RetrievalScore(metric="rank", raw=1, direction="lower"))
    packet = prepare_evidence_packet([observed], max_items=1, snippet_char_limit=40, evidence_char_budget=100)
    response = finalize_answer(text_document(packet[0].excerpt, basis="excerpt", refs=[packet[0].id]), packet)
    case = BenchmarkCase(case_id="budgeted-docs", category="docs_only", query="설명", require_official_citation=True)

    assert packet[0].id != observed.id
    assert score_citation_traceability(case=case, response=response, observed_hits=[hit], called_tools=["tavily_search"]) == 1.0
    assert score_reference_coverage(case=case, response=response, observed_hits=[hit]) == 1.0


@given(start=st.integers(min_value=0, max_value=50), length=st.integers(min_value=1, max_value=20))
def test_only_text_ranges_fully_inside_observed_selection_are_traceable(start: int, length: int) -> None:
    """Every cited character must belong to the original observed selection."""
    source = source_evidence(text="0123456789" * 10)
    observed = build_evidence(snapshot=source.snapshot, element=source.element, start=20, end=40)
    citation = build_evidence(snapshot=source.snapshot, element=source.element, start=start, end=start + length)
    expected = float(set(range(start, start + length)).issubset(range(20, 40)))
    assert citation_score(observed, citation) == expected


@pytest.mark.parametrize("change", [
    {"element_id": "another-element"},
    {"text": "a different original text"},
    {"anchors": [SourceAnchor(kind="page", page_no=4)]},
])
def test_subrange_cannot_switch_the_canonical_source_element(change: dict) -> None:
    """Matching source versions do not allow changed element identity, text, or anchors."""
    observed = source_evidence(text="the captured original text")
    changed_element = observed.element.model_copy(update=change)
    citation = build_evidence(snapshot=observed.snapshot, element=changed_element, start=1, end=8)
    assert citation_score(observed, citation) == 0.0


@pytest.mark.parametrize("selected_cells, expected", [
    (["heading"], 1.0),
    (["value-1"], 1.0),
    (["heading", "value-1"], 1.0),
    (["value-1", "value-2"], 0.0),
    (["value-2"], 0.0),
])
def test_table_citations_require_cells_within_the_observed_cell_selection(selected_cells: list[str], expected: float) -> None:
    """A logical cell subset is traceable without inventing page coordinates."""
    source = source_evidence(text="Setting\n3\n5")
    element = DocumentElement(element_id="table", kind="table", table=TableData(cells=[
        TableCell(cell_id="heading", row=0, col=0, text="Setting", is_header=True),
        TableCell(cell_id="value-1", row=1, col=0, text="3"),
        TableCell(cell_id="value-2", row=2, col=0, text="5"),
    ]))
    observed = build_evidence(snapshot=source.snapshot, element=element, cell_ids=["heading", "value-1"])
    citation = build_evidence(snapshot=source.snapshot, element=element, cell_ids=selected_cells)
    assert citation_score(observed, citation) == expected


def test_element_text_position_is_traceable_without_a_page_anchor() -> None:
    """Snapshot, element, and text offsets still identify a source with no physical location."""
    source = source_evidence(text="source without page coordinates")
    element = source.element.model_copy(update={"anchors": []})
    observed = build_evidence(snapshot=source.snapshot, element=element)
    citation = build_evidence(snapshot=source.snapshot, element=element, start=7, end=14)
    assert citation_score(observed, citation) == 1.0


def citation_score(observed, citation) -> float:
    response = finalize_answer(text_document(citation.excerpt, basis="excerpt", refs=[citation.id]), [citation])
    case = BenchmarkCase(case_id="selection-docs", category="docs_only", query="source", require_official_citation=True)
    hit = SearchHit(evidence=observed, rank=1, score=RetrievalScore(metric="rank", raw=1, direction="lower"))
    return score_citation_traceability(case=case, response=response, observed_hits=[hit], called_tools=["tavily_search"])
