"""Build bounded source candidates; semantic approval belongs to Decisions."""

from __future__ import annotations

import re
from collections.abc import Iterator

from src.core.evidence import SearchHit, build_evidence
from src.core.planner_schema import RetrievalRequirement
from src.infra.tools.docs_search.policy import is_allowed_doc_url, normalized_domain_set, result_matches_domains
from src.infra.tools.docs_search.sections import parse_doc_sections, scan_declarations


_WINDOW_CHARS = 3200


def _ranking_terms(query: str, requirement: RetrievalRequirement) -> dict[str, int]:
    # Literal anchors are only hints for ordering. Their absence never rejects
    # a source or establishes whether a requirement is supported.
    terms = {term.casefold(): 1 for term in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}|[가-힣]{2,}", query)}
    terms.update({term.casefold(): 4 for term in requirement.aspects if term})
    terms.update({term.casefold(): 6 for term in requirement.symbols if term})
    return terms


def _literal_score(text: str, terms: dict[str, int]) -> int:
    lowered = text.casefold()
    return sum(weight for term, weight in terms.items() if term in lowered)


def _window_at(text: str, position: int) -> tuple[int, int]:
    start = max(0, min(position - 400, len(text) - _WINDOW_CHARS))
    # Prefer a complete leading line where possible, without changing the text
    # or suggesting that a possibly truncated trailing line is sufficient.
    boundary = text.find("\n", start, position)
    if boundary >= 0:
        start = boundary + 1
    return start, min(len(text), start + _WINDOW_CHARS)


def _uncovered_chars(span: tuple[int, int], selected: list[tuple[int, int]]) -> int:
    """Count new source positions, independent of the text's meaning."""
    start, end = span
    remaining = end - start
    cursor = start
    for other_start, other_end in sorted(selected):
        overlap_start, overlap_end = max(cursor, other_start), min(end, other_end)
        if overlap_end > overlap_start:
            remaining -= overlap_end - overlap_start
            cursor = overlap_end
    return remaining


def _source_candidates(
    hits: list[SearchHit], terms: dict[str, int], symbols: set[str], limit: int, preserved_evidence_ids: set[str],
) -> Iterator[SearchHit]:
    first = hits[0]
    evidence = first.evidence
    text = evidence.element.text
    if not text:
        return

    # Previous selections keep their precise identity; a new source's first
    # slot may instead use a focused window so long pages do not spend their
    # whole allowance on introductory material.
    originals = dedupe_docs_hits([
        hit for hit in hits if 0 < len(hit.evidence.excerpt) <= _WINDOW_CHARS
    ])
    ranges = {(0, min(len(text), _WINDOW_CHARS))}
    target_declarations = []
    target_sections = []
    if len(text) > _WINDOW_CHARS:
        declarations = scan_declarations(text)
        sections = parse_doc_sections(text, title=evidence.snapshot.title, url=evidence.snapshot.source_uri,
                                      declarations=declarations)
        target_declarations = [item for item in declarations if item.name.casefold() in symbols]
        target_sections = [item for item in sections if item.owner.casefold() in symbols]
        for section in sections:
            # Ownership and section names may guide where a window starts, but
            # cannot exclude other APIs or unrecognized prose on the page.
            ranges.add(_window_at(text, section.start))
        # Structural identity is a soft ordering signal, not an approval gate.
        # Start at the actual definition so earlier APIs do not consume the
        # window that must carry this API's following parameter descriptions.
        for item in [*target_declarations, *target_sections]:
            ranges.add((item.start, min(len(text), item.start + _WINDOW_CHARS)))
        for term in terms:
            for match in re.finditer(re.escape(term), text, flags=re.I):
                ranges.add(_window_at(text, match.start()))
    original_ranges = {
        (hit.evidence.selection.start, hit.evidence.selection.end if hit.evidence.selection.end is not None else len(text)): hit
        for hit in originals
    }
    ranges.update(original_ranges)
    preserved_ranges = {span for span, hit in original_ranges.items() if hit.evidence.id in preserved_evidence_ids}
    literal_scores = {span: _literal_score(text[span[0]:span[1]], terms) for span in ranges}
    structure_scores = {span: (
        3 if any(span[0] == item.start and item.end <= span[1] for item in target_declarations)
        else 2 if any(span[0] == item.start for item in target_sections)
        else 1 if any(item.start <= span[0] < item.end for item in target_sections)
        else 0
    ) for span in ranges}
    # Materialize only bounded windows. Building every possible anchor first
    # would repeatedly copy/hash the full captured source for discarded ranges.
    selected_spans: list[tuple[int, int]] = []
    while ranges and len(selected_spans) < limit:
        remaining_preserved = ranges & preserved_ranges
        if remaining_preserved:
            span = max(remaining_preserved, key=lambda item: (literal_scores[item], -item[0], -item[1]))
        elif not selected_spans:
            span = max(ranges, key=lambda item: (structure_scores[item], literal_scores[item], -item[0], -item[1]))
        else:
            # Nearby anchors often yield almost identical windows. Use actual
            # new positions to keep later sections available to Decisions,
            # instead of repeatedly paying to judge the same opening text.
            span = max(ranges, key=lambda item: (
                _uncovered_chars(item, selected_spans), structure_scores[item], literal_scores[item], -item[0], -item[1],
            ))
            if _uncovered_chars(span, selected_spans) == 0:
                break
        ranges.remove(span)
        selected_spans.append(span)
        start, end = span
        yield original_ranges.get((start, end)) or first.model_copy(update={"evidence": build_evidence(
            snapshot=evidence.snapshot, element=evidence.element, start=start, end=end,
        )})


def select_docs_candidates(
    query: str, hits: list[SearchHit], requirement: RetrievalRequirement, *,
    k: int, allowed_domains: list[str], preferred_snapshot_ids: set[str] | None = None,
    preserved_evidence_ids: set[str] | None = None,
) -> list[SearchHit]:
    """Return source-balanced raw ranges, without any semantic sufficiency label.

    HTTPS, official documentation paths and requested library domains are hard
    constraints. Topic, version, owner and prose recognition are not filters.
    """
    limit = min(20, max(6, k * 3))
    domains = normalized_domain_set(allowed_domains)
    terms = _ranking_terms(query, requirement)
    symbols = {symbol.casefold() for symbol in requirement.symbols}
    sources: dict[tuple[str, str], list[SearchHit]] = {}
    for hit in dedupe_docs_hits(hits):
        evidence = hit.evidence
        if (evidence.route != "docs" or not is_allowed_doc_url(evidence.snapshot.source_uri)
                or not result_matches_domains(evidence.snapshot.source_uri, domains)):
            continue
        sources.setdefault((evidence.snapshot.snapshot_id, evidence.element.element_id), []).append(hit)

    groups = list(sources.values())
    groups.sort(key=lambda group: (
        group[0].evidence.snapshot.snapshot_id not in (preferred_snapshot_ids or set()),
        -_literal_score(group[0].evidence.snapshot.title + " " + group[0].evidence.snapshot.source_uri, terms),
        -float(group[0].score.normalized or 0),
        group[0].evidence.snapshot.source_uri,
    ))
    # Every admitted source gets a window before one rich source
    # consumes remaining slots with multiple headings or literal matches.
    pools = [_source_candidates(group, terms, symbols, limit, preserved_evidence_ids or set()) for group in groups[:limit]]
    selected: list[SearchHit] = []
    while len(selected) < limit and pools:
        for pool in pools[:]:
            hit = next(pool, None)
            if hit is None:
                pools.remove(pool)
                continue
            selected.append(hit)
            if len(selected) == limit:
                break
    return [hit.model_copy(update={"rank": index}) for index, hit in enumerate(selected, start=1)]


def dedupe_docs_hits(items: list[SearchHit]) -> list[SearchHit]:
    """Deduplicate exact selections without concatenating source excerpts."""
    by_evidence: dict[str, SearchHit] = {}
    for hit in items:
        current = by_evidence.get(hit.evidence.id)
        score = hit.score.normalized
        current_score = current.score.normalized if current else None
        if current is None or (score is not None and (current_score is None or score > current_score)):
            by_evidence[hit.evidence.id] = hit
    return [hit.model_copy(update={"rank": index}) for index, hit in enumerate(by_evidence.values(), start=1)]
