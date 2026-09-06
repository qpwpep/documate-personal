from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urlparse

from src.core.answer_schema import AnswerResponse, export_answer_text, iter_content_units
from src.core.domain_docs import DEFAULT_DOCS
from src.core.evidence import EvidenceRef, SearchHit
from .config_models import BenchmarkCase


_FAILURE_TEXT_PATTERNS = [
    r"agent execution failed",
    r"request timeout",
    r"unexpected error",
]
_HANGUL_PATTERN = re.compile(r"[가-힣]")
_COMPARISON_MARKERS = (
    "비교",
    "차이",
    "반면",
    "다르",
    "contrast",
    "compare",
    "difference",
    "however",
    "whereas",
)
_ALLOWED_OFFICIAL_DOMAINS = set()


def score_tool_choice(
    case: BenchmarkCase,
    called_tools: list[str],
    *,
    slack_delivery_required: bool = False,
    slack_delivery_status: str = "not_applicable",
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
            if tool_name not in called:
                continue
            if tool_name == "slack_notify" and slack_delivery_required:
                if slack_delivery_status == "success":
                    matched_expected += 1
                continue
            matched_expected += 1
        expected_score = matched_expected / len(expected)

    forbidden_penalty = 0.0
    if forbidden:
        forbidden_penalty = len(forbidden.intersection(called)) / len(forbidden)

    return max(0.0, expected_score * (1.0 - forbidden_penalty))


def score_answer_quality(
    case: BenchmarkCase,
    response_text: str,
    observed_hits: list[SearchHit],
    *,
    synthesis_mode: str | None = None,
) -> float:
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

    copy_penalty = _copy_penalty(text, observed_hits)
    if copy_penalty > 0.0:
        quality = max(0.0, quality - copy_penalty)

    if case.category == "hybrid" and not _hybrid_comparison_present(text):
        quality = min(quality, 0.25)

    if case.category in {"docs_only", "hybrid"} and synthesis_mode == "deterministic_grounded_direct":
        quality = min(quality, 0.2)

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


def _traceable_refs(response: AnswerResponse, observed_hits: list[SearchHit]) -> set[str]:
    used = {ref for _, unit in iter_content_units(response.content) for ref in unit.refs}
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
    traceable = _traceable_refs(response, observed_hits)
    route_tools = {"docs": "tavily_search", "upload": "upload_search"}
    valid_routes = {
        citation.evidence.route
        for citation in response.citations
        if citation.evidence.id in traceable
        and (
            citation.evidence.snapshot.source_type != "official"
            or _is_valid_official_source(citation.evidence.snapshot.source_uri)
        )
    }
    used = {ref for _, unit in iter_content_units(response.content) for ref in unit.refs}
    coverage = len(used.intersection(traceable)) / len(used) if used else 0.0
    route_coverage = sum(
        route in valid_routes and route_tools[route] in called_tools for route in required_routes
    ) / len(required_routes)
    return min(coverage, route_coverage)


def score_reference_coverage(
    *,
    case: BenchmarkCase | None = None,
    response: AnswerResponse | None,
    observed_hits: list[SearchHit],
    validator_reason: str | None = None,
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
    traceable = _traceable_refs(response, observed_hits)
    return sum(bool(unit.refs) and all(ref in traceable for ref in unit.refs) for unit in units) / len(units)


def score_format_language(
    *,
    case: BenchmarkCase,
    runtime_errors: list[str],
    response_errors: list[str],
    judge_errors: list[str],
    response_text: str,
) -> float:
    if runtime_errors or response_errors:
        return 0.0

    text = (response_text or "").strip()
    if not text:
        return 0.0
    if _contains_any_pattern(text, _FAILURE_TEXT_PATTERNS):
        return 0.0
    if _contains_hangul(case.query) and not _contains_hangul(text):
        return 0.0
    if any(str(error).startswith("invalid_eval:") for error in judge_errors):
        return 0.0
    return 1.0


def compute_rule_scores(
    *,
    case: BenchmarkCase,
    response: AnswerResponse | None,
    called_tools: list[str],
    observed_hits: list[SearchHit],
    runtime_errors: list[str],
    response_errors: list[str],
    judge_errors: list[str],
    validator_reason: str | None = None,
    synthesis_mode: str | None = None,
    slack_delivery_required: bool = False,
    slack_delivery_status: str = "not_applicable",
) -> dict[str, float]:
    response_text = export_answer_text(response) if response is not None else ""
    return {
        "answer_quality": score_answer_quality(case, response_text, observed_hits, synthesis_mode=synthesis_mode),
        "reference_coverage": score_reference_coverage(case=case, response=response, observed_hits=observed_hits, validator_reason=validator_reason),
        "citation_traceability": score_citation_traceability(case=case, response=response, observed_hits=observed_hits, called_tools=called_tools),
        "tool_choice": score_tool_choice(case, called_tools, slack_delivery_required=slack_delivery_required, slack_delivery_status=slack_delivery_status),
        "format_language": score_format_language(case=case, runtime_errors=runtime_errors, response_errors=response_errors, judge_errors=judge_errors, response_text=response_text),
    }


def tool_confusion_counts(case: BenchmarkCase, called_tools: list[str]) -> tuple[int, int, int]:
    expected = set(case.expected_tools)
    forbidden = set(case.forbidden_tools)
    called = set(called_tools)

    tp = len(expected.intersection(called))
    fn = len(expected.difference(called))
    fp = 0
    if forbidden:
        fp += len(forbidden.intersection(called))
    if expected:
        fp += len(called.difference(expected).difference(forbidden))
    return tp, fp, fn


def _contains_any_pattern(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def _normalize_domain(url_or_domain: str) -> str:
    parsed = urlparse(url_or_domain if "://" in url_or_domain else f"https://{url_or_domain}")
    domain = (parsed.netloc or parsed.path).strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


for _default_doc in DEFAULT_DOCS.values():
    _ALLOWED_OFFICIAL_DOMAINS.add(_normalize_domain(_default_doc))


def _is_valid_official_source(url_or_path: str) -> bool:
    parsed = urlparse(str(url_or_path or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return False
    return _normalize_domain(parsed.netloc) in _ALLOWED_OFFICIAL_DOMAINS


def _contains_hangul(text: str) -> bool:
    return bool(_HANGUL_PATTERN.search(str(text or "")))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _copy_penalty(response_text: str, observed_hits: list[SearchHit]) -> float:
    normalized_response = _normalize_text(response_text)
    if not normalized_response:
        return 0.0
    longest_match = 0
    for hit in observed_hits:
        snippet = _normalize_text(hit.evidence.excerpt)
        if len(snippet) < 48:
            continue
        if snippet and snippet in normalized_response:
            longest_match = max(longest_match, len(snippet))
    if longest_match >= 120:
        return 0.65
    if longest_match >= 72:
        return 0.45
    return 0.0


def _hybrid_comparison_present(response_text: str) -> bool:
    normalized = _normalize_text(response_text)
    return any(marker in normalized for marker in _COMPARISON_MARKERS)
