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


def _claims(text: str, *, native_code_context: bool) -> list[tuple[int, int]]:
    # A literal string may itself contain 'line 29'; that is quoted source data.
    prose = _INLINE.sub("", text).replace("**", "").replace("__", "")
    source_context = bool(_FILE.search(text) or _SOURCE_CONTEXT.search(prose))
    if not (source_context or native_code_context):
        return []
    matches = [*_ENGLISH_LINE.finditer(prose), *_KOREAN_LINE.finditer(prose), *_LABELED_LINE.finditer(prose)]
    return list(dict.fromkeys(
        (int(match["start"]), int(match["end"] or match["start"]))
        for match in matches if not _ROW_COUNT.match(prose[match.end():])
        and not _ROW_SUBJECT.search(prose[:match.start()])
    ))


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


def _literals_for_unit(
    unit: ContentUnit, block_index: int, document: AnswerDocument, sources: list[EvidenceRef],
) -> list[str]:
    inline = [match[1] for match in _INLINE.finditer(unit.text)
              if not _FILE.fullmatch(match[1]) and any(match[1] in source.excerpt for source in sources)]
    if inline:
        return inline
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
        literals = _literals_for_unit(unit, block_index, document, sources)
        native_code_context = any(
            anchor.kind in {"code", "notebook"} and literal in source.excerpt
            for source in sources for anchor in source.element.anchors for literal in literals
        )
        claims = _claims(unit.text, native_code_context=native_code_context)
        if not claims:
            continue
        literal_locations = [{location for source in sources for location in _occurrences(literal, source)}
                             for literal in literals]
        locations = set.intersection(*literal_locations) if literal_locations else set()
        if not locations:
            failures[path] = ["source_location_unverifiable"]
        elif any(claim not in locations for claim in claims):
            failures[path] = ["source_location_mismatch"]
    return failures
