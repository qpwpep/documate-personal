"""Build bounded source candidates; semantic approval belongs to Decisions."""

from __future__ import annotations

import re
from collections.abc import Iterator
from urllib.parse import unquote, urlparse

from src.core.evidence import SearchHit, build_evidence
from src.core.planner_schema import RetrievalRequirement
from src.infra.tools.docs_search.policy import (
    DOCS_WINDOW_CHARS, is_allowed_doc_url, is_current_stable_doc_url, normalized_domain_set, result_matches_domains,
)
from src.infra.tools.docs_search.sections import document_symbol_names, parse_doc_sections, scan_declarations


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
    start = max(0, min(position - 400, len(text) - DOCS_WINDOW_CHARS))
    # Prefer a complete leading line where possible, without changing the text
    # or suggesting that a possibly truncated trailing line is sufficient.
    boundary = text.find("\n", start, position)
    if boundary >= 0:
        start = boundary + 1
    return start, min(len(text), start + DOCS_WINDOW_CHARS)


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
        hit for hit in hits if 0 < len(hit.evidence.excerpt) <= DOCS_WINDOW_CHARS
    ])
    ranges = {(0, min(len(text), DOCS_WINDOW_CHARS))}
    target_declarations = []
    target_sections = []
    if len(text) > DOCS_WINDOW_CHARS:
        declarations = scan_declarations(text)
        sections = parse_doc_sections(text, title=evidence.snapshot.title, url=evidence.snapshot.source_uri,
                                      declarations=declarations)
        matching_symbols = document_symbol_names(symbols, title=evidence.snapshot.title,
                                                url=evidence.snapshot.source_uri)
        target_declarations = [item for item in declarations if item.name.casefold() in matching_symbols]
        target_sections = [item for item in sections if item.owner.casefold() in matching_symbols]
        for section in sections:
            # Ownership and section names may guide where a window starts, but
            # cannot exclude other APIs or unrecognized prose on the page.
            ranges.add(_window_at(text, section.start))
        # Structural identity is a soft ordering signal, not an approval gate.
        # Start at the actual definition so earlier APIs do not consume the
        # window that must carry this API's following parameter descriptions.
        for item in [*target_declarations, *target_sections]:
            ranges.add((item.start, min(len(text), item.start + DOCS_WINDOW_CHARS)))
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
    With no requested version, moving stable aliases receive the first slots.
    Explicit versions prioritize the named API's matching documentation first.
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
    source_priorities = {}
    version_pattern = (re.compile(rf"(?<![\w.-]){re.escape(requirement.version)}(?![\w.-])", re.I)
                       if requirement.version else None)
    symbol_patterns = [re.compile(rf"(?<![\w.-]){re.escape(symbol)}(?![\w.-])", re.I)
                       for symbol in requirement.symbols]
    for group in groups:
        snapshot = group[0].evidence.snapshot
        path = unquote(urlparse(snapshot.source_uri).path)
        symbol_context = snapshot.title + " " + re.sub(r"\.html?$", "", path, flags=re.I)
        matches_symbol = any(pattern.search(symbol_context) for pattern in symbol_patterns)
        if version_pattern is None:
            source_priorities[snapshot.snapshot_id] = 2
            if is_current_stable_doc_url(snapshot.source_uri):
                source_priorities[snapshot.snapshot_id] = (
                    0 if matches_symbol and snapshot.parser == "official_html" else 1
                )
            continue
        # Titles and paths can guide allocation, never approve a version or
        # subject. Query strings and fragments cannot establish either identity.
        source_priorities[snapshot.snapshot_id] = 2
        if version_pattern.search(snapshot.title) or version_pattern.search(path):
            source_priorities[snapshot.snapshot_id] = 0 if matches_symbol else 1
    groups.sort(key=lambda group: (
        source_priorities[group[0].evidence.snapshot.snapshot_id],
        group[0].evidence.snapshot.snapshot_id not in (preferred_snapshot_ids or set()),
        -_literal_score(group[0].evidence.snapshot.title + " " + group[0].evidence.snapshot.source_uri, terms),
        -float(group[0].score.normalized or 0),
        group[0].evidence.snapshot.source_uri,
    ))
    groups = groups[:limit]
    # Exhaust the preferred tier first so other APIs/versions cannot displace
    # its later parameters and examples. Preserve round-robin within each tier.
    batches = [[group for group in groups if source_priorities[group[0].evidence.snapshot.snapshot_id] == priority]
               for priority in range(3)]
    selected: list[SearchHit] = []
    for batch in batches:
        if len(selected) == limit:
            break
        # Balance sources within each tier before selecting more windows.
        pools = [_source_candidates(group, terms, symbols, limit, preserved_evidence_ids or set()) for group in batch]
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
