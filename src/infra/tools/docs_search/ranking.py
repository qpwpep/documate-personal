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
from src.infra.tools.docs_search.sections import (
    DocDeclaration, DocSection, definition_sections, document_symbol_names, parse_doc_sections,
    resolve_document_symbol, scan_declarations,
)


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


def _window_at(text: str, position: int, *, leading_chars: int = 400) -> tuple[int, int]:
    start = position if leading_chars == 0 else max(0, min(position - leading_chars, len(text) - DOCS_WINDOW_CHARS))
    # A trailing condition can be the only part that answers the request. Use
    # available leading context to finish that line without dropping the anchor
    # or exceeding the window limit; otherwise stop at the previous whole line.
    boundary = text.find("\n", start, position)
    if boundary >= 0:
        start = boundary + 1
    end = min(len(text), start + DOCS_WINDOW_CHARS)
    if end == len(text) or text[end - 1:end] == "\n":
        return start, end
    following = text.find("\n", end, min(len(text), position + DOCS_WINDOW_CHARS))
    if following >= 0:
        shifted_start = max(start, following + 1 - DOCS_WINDOW_CHARS)
        shifted = text.find("\n", shifted_start - 1, position)
        return shifted + 1 if shifted >= 0 else shifted_start, following + 1
    previous = text.rfind("\n", start, end)
    return start, previous + 1 if previous >= position else end


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
    structure: tuple[list[DocDeclaration], list[DocSection], list[DocSection]] | None = None,
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
    ranges = {_window_at(text, 0, leading_chars=0)}
    target_declarations = []
    target_sections = []
    definitions = []
    if len(text) > DOCS_WINDOW_CHARS:
        if structure is None:
            declarations = scan_declarations(text)
            sections = parse_doc_sections(text, title=evidence.snapshot.title, url=evidence.snapshot.source_uri,
                                          declarations=declarations)
            definitions = definition_sections(text, declarations=declarations, sections=sections)
        else:
            declarations, sections, definitions = structure
        matching_symbols = document_symbol_names(symbols, title=evidence.snapshot.title,
                                                url=evidence.snapshot.source_uri)
        target_declarations = [item for item in declarations if item.name.casefold() in matching_symbols]
        target_sections = [item for item in sections if document_symbol_names(
            [item.owner], title=evidence.snapshot.title, url=evidence.snapshot.source_uri,
        ) & matching_symbols]
        for section in sections:
            # Ownership and section names may guide where a window starts, but
            # cannot exclude other APIs or unrecognized prose on the page.
            ranges.add(_window_at(text, section.start))
        # Structural identity is a soft ordering signal, not an approval gate.
        # Start at the actual definition so earlier APIs do not consume the
        # window that must carry this API's following parameter descriptions.
        for item in [*target_declarations, *target_sections]:
            ranges.add(_window_at(text, item.start, leading_chars=0))
        for term in terms:
            for match in re.finditer(re.escape(term), text, flags=re.I):
                ranges.add(_window_at(text, match.start()))
    original_ranges = {
        (hit.evidence.selection.start, hit.evidence.selection.end if hit.evidence.selection.end is not None else len(text)): hit
        for hit in originals
    }
    preserved_ranges = {span for span, hit in original_ranges.items() if hit.evidence.id in preserved_evidence_ids}
    ranges.update(preserved_ranges)
    for start, end in original_ranges:
        if end == len(text) or text[end - 1:end] == "\n":
            ranges.add((start, end))
        else:
            ranges.add(_window_at(text, start, leading_chars=0))
    literal_scores = {span: _literal_score(text[span[0]:span[1]], terms) for span in ranges}
    # A definition with matching structural ownership is stronger than a bare
    # signature mentioned in a reference table or another API's notes.
    owned_declaration_starts = {item.start for item in target_sections}
    if definitions:
        owned_declaration_starts.intersection_update(item.start for item in definitions)
    structure_scores = {span: (
        3 + (span[0] in owned_declaration_starts)
        if any(span[0] == item.start and item.end <= span[1] for item in target_declarations)
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
            # Compare the uncovered fraction: a complete definition window
            # may end before the character limit to preserve its final line.
            # That must not rank below every longer, unrelated window.
            span = max(ranges, key=lambda item: (
                _uncovered_chars(item, selected_spans) / (item[1] - item[0]),
                structure_scores[item], literal_scores[item], -item[0], -item[1],
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
    structures: dict[tuple[str, str], tuple[list[DocDeclaration], list[DocSection], list[DocSection]]] = {}
    version_pattern = (re.compile(rf"(?<![\w.-]){re.escape(requirement.version)}(?![\w.-])", re.I)
                       if requirement.version else None)
    symbol_patterns = [re.compile(rf"(?<![\w.-]){re.escape(symbol)}(?![\w.-])", re.I)
                       for symbol in requirement.symbols]
    for group in groups:
        evidence = group[0].evidence
        snapshot = evidence.snapshot
        path = unquote(urlparse(snapshot.source_uri).path)
        symbol_context = snapshot.title + " " + re.sub(r"\.html?$", "", path, flags=re.I)
        identity = resolve_document_symbol(title=snapshot.title, url=snapshot.source_uri).casefold()
        # Module references identify their namespace rather than every member
        # in their title. Reserve the captured module's later windows before
        # unrelated stable pages spend the source budget.
        matches_symbol = any(pattern.search(symbol_context) for pattern in symbol_patterns) or bool(
            identity and any(symbol.casefold().startswith(identity + ".") for symbol in requirement.symbols)
        )
        if not matches_symbol and symbols:
            declarations = scan_declarations(evidence.element.text)
            sections = parse_doc_sections(evidence.element.text, title=snapshot.title,
                url=snapshot.source_uri, declarations=declarations)
            definitions = definition_sections(evidence.element.text, declarations=declarations, sections=sections)
            structures[(snapshot.snapshot_id, evidence.element.element_id)] = (declarations, sections, definitions)
            # Module and builtin reference titles need not list every member.
            # A source-owned, self-linked declaration identifies the API;
            # an index entry or a call inside an example does not.
            matches_symbol = bool(
                document_symbol_names((item.owner for item in definitions), title=snapshot.title, url=snapshot.source_uri)
                & document_symbol_names(symbols, title=snapshot.title, url=snapshot.source_uri)
            )
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
        pools = [_source_candidates(group, terms, symbols, limit, preserved_evidence_ids or set(),
            structures.get((group[0].evidence.snapshot.snapshot_id, group[0].evidence.element.element_id)))
            for group in batch]
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
