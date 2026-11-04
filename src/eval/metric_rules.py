from __future__ import annotations

from src.core.answer_schema import AnswerResponse, export_answer_text, iter_content_units
from src.core.contracts.outcome import TurnResult
from src.core.evidence import EvidenceRef, SearchHit
from src.infra.tools.docs_search.policy import is_allowed_doc_url
from .config_models import BenchmarkCase
from .result_models import EvidenceAssessment


def score_tool_choice(
    case: BenchmarkCase,
    called_tools: list[str],
    *,
    slack_delivery_required: bool = False,
    slack_delivery_status: str = "not_applicable",
    save_outcome_verified: bool = False,
) -> float:
    expected = set(case.expected_tools)
    forbidden = set(case.forbidden_tools)
    called = set(called_tools)

    if not expected and not forbidden:
        return 1.0

    expected_score = 1.0
    if expected:
        matched_expected = 0
        for tool_name in expected:
            if tool_name == "save_text" and save_outcome_verified:
                matched_expected += 1
                continue
            if tool_name not in called:
                continue
            if tool_name == "slack_notify" and slack_delivery_required:
                if slack_delivery_status == "success":
                    matched_expected += 1
                continue
            matched_expected += 1
        expected_score = matched_expected / len(expected)

    # Prohibited execution is assessed independently by tool_policy. Quality
    # scores never decide whether a mandatory policy violation is acceptable.
    return expected_score


def score_answer_quality(
    case: BenchmarkCase,
    response_text: str,
    observed_hits: list[SearchHit],
) -> float:
    """Score explicit text constraints, not semantic proxies.

    Evidence category, comparison words, and retrieval chunk boundaries cannot
    establish whether the user's task was fulfilled. The semantic judge checks
    the query and oracle, including whether a comparison or verbatim quotation
    was requested.
    """
    text = response_text or ""
    if not text.strip():
        return 0.0

    include_score = 1.0
    if case.must_include:
        include_hits = sum(1 for needle in case.must_include if needle.lower() in text.lower())
        include_score = include_hits / len(case.must_include)

    exclude_score = 1.0
    if case.must_not_include:
        exclude_violations = sum(1 for needle in case.must_not_include if needle.lower() in text.lower())
        exclude_score = 1.0 - (exclude_violations / len(case.must_not_include))

    quality = max(0.0, min(1.0, (include_score + exclude_score) / 2.0))

    return max(0.0, min(1.0, quality))


def _source_selection_contains(observed: EvidenceRef, cited: EvidenceRef) -> bool:
    """A prompt may narrow a hit; it cannot change its source or expand its range."""
    if observed.snapshot != cited.snapshot or observed.element != cited.element:
        return False
    observed_cells = set(observed.selection.cell_ids)
    cited_cells = set(cited.selection.cell_ids)
    if observed_cells or cited_cells:
        return bool(cited_cells) and cited_cells.issubset(observed_cells)
    observed_end = len(observed.element.text) if observed.selection.end is None else observed.selection.end
    cited_end = len(cited.element.text) if cited.selection.end is None else cited.selection.end
    return observed.selection.start <= cited.selection.start < cited_end <= observed_end


def _traceable_refs(response: AnswerResponse, observed_hits: list[SearchHit],
                    evidence_scope: EvidenceAssessment | None = None) -> set[str]:
    used = {ref for _, unit in iter_content_units(response.content) for ref in unit.refs}
    if evidence_scope is not None:
        if evidence_scope.status != "complete":
            return set()
        verified = {item.id: item for item in evidence_scope.verified_evidence}
        return {citation.evidence.id for citation in response.citations
                if citation.evidence.id in used and verified.get(citation.evidence.id) == citation.evidence}
    return {
        citation.evidence.id
        for citation in response.citations
        if citation.evidence.id in used
        and any(_source_selection_contains(hit.evidence, citation.evidence) for hit in observed_hits)
    }


def score_citation_traceability(
    *,
    case: BenchmarkCase,
    response: AnswerResponse | None,
    observed_hits: list[SearchHit],
    called_tools: list[str],
    evidence_scope: EvidenceAssessment | None = None,
) -> float:
    required_routes = []
    if case.require_official_citation:
        required_routes.append("docs")
    if case.require_local_citation:
        required_routes.append("upload")
    if not required_routes:
        return 1.0
    if response is None:
        return 0.0
    traceable = _traceable_refs(response, observed_hits, evidence_scope)
    route_tools = {"docs": "tavily_search", "upload": "upload_search"}
    valid_refs = {
        citation.evidence.id
        for citation in response.citations
        if citation.evidence.id in traceable
        and (
            citation.evidence.snapshot.source_type != "official"
            or is_allowed_doc_url(citation.evidence.snapshot.source_uri)
        )
    }
    valid_routes = {
        citation.evidence.route for citation in response.citations
        if citation.evidence.id in valid_refs
    }
    used = {ref for _, unit in iter_content_units(response.content) for ref in unit.refs}
    coverage = len(used.intersection(valid_refs)) / len(used) if used else 0.0
    route_coverage = sum(
        route in valid_routes and (evidence_scope is not None or route_tools[route] in called_tools)
        for route in required_routes
    ) / len(required_routes)
    return min(coverage, route_coverage)


def score_reference_coverage(
    *,
    case: BenchmarkCase | None = None,
    response: AnswerResponse | None,
    observed_hits: list[SearchHit],
    validator_reason: str | None = None,
    evidence_scope: EvidenceAssessment | None = None,
) -> float:
    """Measure reference coverage only; semantic groundedness belongs to the judge."""
    if response is None or not export_answer_text(response).strip():
        return 0.0
    if case is not None and case.category == "tool_action":
        if not case.require_official_citation and not case.require_local_citation:
            return 1.0
    units = [unit for _, unit in iter_content_units(response.content) if unit.basis != "interaction"]
    if not units or validator_reason == "no_evidence":
        return 0.0
    traceable = _traceable_refs(response, observed_hits, evidence_scope)
    return sum(bool(unit.refs) and all(ref in traceable for ref in unit.refs) for unit in units) / len(units)


def score_format_language(
    *,
    case: BenchmarkCase,
    runtime_errors: list[str],
    response_errors: list[str],
    response_text: str,
) -> float:
    """Check response availability without interpreting the answer as diagnostics.

    This compatibility metric uses structured runtime/response errors. The
    semantic judge evaluates requested structure and language from the query
    and oracle, including literal copies and explicitly requested languages.
    """
    if runtime_errors or response_errors:
        return 0.0
    return float(bool((response_text or "").strip()))


def compute_rule_scores(
    *,
    case: BenchmarkCase,
    response: AnswerResponse | None,
    turn_result: TurnResult | None = None,
    called_tools: list[str],
    observed_hits: list[SearchHit],
    runtime_errors: list[str],
    response_errors: list[str],
    validator_reason: str | None = None,
    slack_delivery_required: bool = False,
    slack_delivery_status: str = "not_applicable",
    evidence_scope: EvidenceAssessment | None = None,
    save_outcome_verified: bool = False,
) -> dict[str, float]:
    response_text = export_answer_text(response) if response is not None else ""
    if turn_result is not None and turn_result.status == "needs_input":
        response_text = turn_result.message
    return {
        "answer_quality": score_answer_quality(case, response_text, observed_hits),
        "reference_coverage": score_reference_coverage(case=case, response=response, observed_hits=observed_hits, validator_reason=validator_reason, evidence_scope=evidence_scope),
        "citation_traceability": score_citation_traceability(case=case, response=response, observed_hits=observed_hits, called_tools=called_tools, evidence_scope=evidence_scope),
        "tool_choice": score_tool_choice(case, called_tools, slack_delivery_required=slack_delivery_required, slack_delivery_status=slack_delivery_status, save_outcome_verified=save_outcome_verified),
        "format_language": score_format_language(case=case, runtime_errors=runtime_errors, response_errors=response_errors, response_text=response_text),
    }


def tool_confusion_counts(case: BenchmarkCase, called_tools: list[str], *, save_outcome_verified: bool = False) -> tuple[int, int, int]:
    expected = set(case.expected_tools)
    forbidden = set(case.forbidden_tools)
    called = set(called_tools)
    if save_outcome_verified and "save_text" in expected:
        called.add("save_text")

    tp = len(expected.intersection(called))
    fn = len(expected.difference(called))
    fp = 0
    if forbidden:
        fp += len(forbidden.intersection(called))
    if expected:
        fp += len(called.difference(expected).difference(forbidden))
    return tp, fp, fn
