"""Reject explicit source-line claims that contradict retained physical locations.

This check binds quoted source text to a position; it does not judge the meaning
of a paraphrase or certify the rest of a content unit. Unknown provenance cannot
be turned into a physical line number by a model confidence score.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.core.evidence import EvidenceRef, source_line_records

if TYPE_CHECKING:
    from src.core.answer_schema.models import AnswerDocument, ContentUnit


_INLINE = re.compile(r"`+([^`]+)`+")
_FILE = re.compile(r"[\w.-]+\.(?:py|ipynb|txt|md|csv|json|js|ts|tsx|jsx|java|c|cpp|h)(?![A-Za-z0-9_.])", re.IGNORECASE)
_SOURCE_CONTEXT = re.compile(r"파일|원본|원문|소스|줄\s*번호|행\s*번호|코드(?:는|가|의)|\bsource\b", re.IGNORECASE)
_NUMBER_RANGE = r"(?P<start>\d+)(?:\s*[-~–—]\s*(?P<end>\d+))?"
_KOREAN_LINE = re.compile(r"(?<![\w\d])" + _NUMBER_RANGE + r"\s*(?:번(?:째)?\s*)?(?:행|줄)")
_ENGLISH_LINE = re.compile(r"\blines?\s*:?\s*" + _NUMBER_RANGE + r"\b", re.IGNORECASE)
_LABELED_LINE = re.compile(r"(?:줄|행)\s*번호\s*[:：]?\s*" + _NUMBER_RANGE)
_ROW_COUNT = re.compile(r"^\s*(?:의\s*)?(?:데이터|레코드|DataFrame|rows?\b)", re.IGNORECASE)
_ROW_SUBJECT = re.compile(r"(?:DataFrame|데이터프레임|데이터|표|결과)(?:은|는|이|가|에는|에|의)?\s*(?:총\s*)?$", re.IGNORECASE)


def _claims(text: str, *, native_code_context: bool) -> list[re.Match[str]]:
    # A literal string may itself contain 'line 29'; that is quoted source data.
    # Mask it without shifting the offsets used to bind a claim to its quote.
    masked = _INLINE.sub(lambda match: " " * len(match[0]), text)
    prose = masked.replace("**", "  ").replace("__", "  ")
    source_context = bool(_FILE.search(text) or _SOURCE_CONTEXT.search(prose))
    if not (source_context or native_code_context):
        return []
    matches = [*_ENGLISH_LINE.finditer(prose), *_KOREAN_LINE.finditer(prose), *_LABELED_LINE.finditer(prose)]
    claims: list[re.Match[str]] = []
    for match in sorted(matches, key=lambda item: (item.start(), -item.end())):
        if claims and match.start() < claims[-1].end():
            continue
        before = masked[:match.start()].replace("**", "").replace("__", "")
        after = masked[match.end():].replace("**", "").replace("__", "")
        if not _ROW_COUNT.match(after) and not _ROW_SUBJECT.search(before):
            claims.append(match)
    return claims


def _occurrences(literal: str, evidence: EvidenceRef) -> set[tuple[int, int]]:
    records = source_line_records(evidence)
    if not literal or not records:
        return set()
    positions: set[tuple[int, int]] = set()
    start = 0
    while (start := evidence.excerpt.find(literal, start)) >= 0:
        absolute_start = evidence.selection.start + start
        absolute_end = absolute_start + len(literal)
        first = next((row["line_number"] for row in records
                      if row["start"] <= absolute_start < row["end"]), None)
        last = next((row["line_number"] for row in records
                     if row["start"] <= absolute_end - 1 < row["end"]), None)
        if first is not None and last is not None:
            positions.add((first, last))
        start += 1
    return positions


def _neighboring_code_literals(
    unit: ContentUnit, block_index: int, document: AnswerDocument, sources: list[EvidenceRef],
) -> list[str]:
    # Only neighboring source code associated with this unit's references can
    # supply the literal. Equally near blocks must agree on their locations;
    # neither document order nor the first textual occurrence resolves ambiguity.
    code = [(abs(index - block_index), block.content.text)
            for index, block in enumerate(document.blocks)
            if block.type == "code" and set(block.content.refs).intersection(unit.refs)
            and any(block.content.text in source.excerpt for source in sources)]
    if not code:
        return []
    distance = min(item[0] for item in code)
    return list(dict.fromkeys(text for near, text in code if near == distance))


def _claim_literals(
    text: str, claims: list[re.Match[str]], inline: list[re.Match[str]], literals: list[str],
) -> list[list[str]]:
    if len(claims) == 1 or not inline or len(set(literals)) == 1:
        return [literals for _ in claims]
    # Distinct claim/quote pairs keep their textual order even if clauses
    # alternate direction. Overlapping pairs such as "lines 1 and 2 both
    # contain `x` and `y`" do not establish that one-to-one binding.
    if len(claims) == len(inline):
        spans = [(min(claim.start(), literal.start()), max(claim.end(), literal.end()))
                 for claim, literal in zip(claims, inline)]
        if all(left[1] <= right[0] for left, right in zip(spans, spans[1:])):
            return [[match[1]] for match in inline]
    # In a sequence of explicit claims, quotes either follow their positions
    # ("line 1 is `x`") or precede them ("`x` is on line 1"). Keep each quote
    # within those claim boundaries; another clause cannot validate its line.
    positions_first = claims[0].start() < inline[0].start()
    bindings: list[list[str]] = []
    for index, claim in enumerate(claims):
        if positions_first:
            start = claim.end()
            end = claims[index + 1].start() if index + 1 < len(claims) else len(text)
        else:
            start = claims[index - 1].end() if index else 0
            end = claim.start()
        bindings.append([match[1] for match in inline if start <= match.start() and match.end() <= end])
    # A shared list of positions has no individual quote for every claim.
    # Retain the conservative check that every literal occurs at every position.
    return bindings if all(bindings) else [literals for _ in claims]


def source_location_issues(document: AnswerDocument, evidence: list[EvidenceRef]) -> dict[str, list[str]]:
    """Return deterministic failures, leaving correct claims for semantic review."""
    # Imported at execution time to keep the schema's validation hook acyclic.
    from src.core.answer_schema.models import iter_content_units

    by_id = {item.id: item for item in evidence}
    failures: dict[str, list[str]] = {}
    for path, unit in iter_content_units(document):
        block_index = int(path.split(".", 1)[0][1:])
        if document.blocks[block_index].type == "code":
            continue
        sources = [by_id[ref] for ref in unit.refs if ref in by_id]
        if any(unit.text == source.excerpt for source in sources):
            continue
        named_files = set(_FILE.findall(unit.text))
        if named_files:
            sources = [source for source in sources if source.snapshot.title in named_files
                       or source.snapshot.source_uri.replace("\\", "/").rsplit("/", 1)[-1] in named_files]
        inline = [match for match in _INLINE.finditer(unit.text)
                  if not _FILE.fullmatch(match[1]) and any(match[1] in source.excerpt for source in sources)]
        literals = ([match[1] for match in inline] if inline
                    else _neighboring_code_literals(unit, block_index, document, sources))
        native_code_context = any(
            anchor.kind in {"code", "notebook"} and literal in source.excerpt
            for source in sources for anchor in source.element.anchors for literal in literals
        )
        claims = _claims(unit.text, native_code_context=native_code_context)
        if not claims:
            continue
        for claim, bound_literals in zip(claims, _claim_literals(unit.text, claims, inline, literals)):
            literal_locations = [{location for source in sources for location in _occurrences(literal, source)}
                                 for literal in bound_literals]
            locations = set.intersection(*literal_locations) if literal_locations else set()
            claimed_lines = (int(claim["start"]), int(claim["end"] or claim["start"]))
            if not locations:
                failures[path] = ["source_location_unverifiable"]
                break
            if claimed_lines not in locations:
                failures[path] = ["source_location_mismatch"]
                break
    return failures
