"""Count displayed prose under explicit sentence quotas without changing its text.

Generated prose uses terminal punctuation (.!?。！？), with whitespace or the end
of a prose group after a full stop. Decimal points and unspaced dotted names are
not terminators. Ambiguous abbreviations must be expanded by the generator.
Markdown syntax, link destinations and inline code are not prose punctuation.
Any remaining prose fragment counts once, so omitting punctuation cannot hide it.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING
import unicodedata

from markdown_it import MarkdownIt

from src.core.answer_schema.models import AnswerDocument, ContentUnit
from src.core.answer_schema.rendering import content_unit_markdown, inline_code_markdown

if TYPE_CHECKING:
    from src.core.request_contracts import SentenceCountRequirement


_MARKDOWN = MarkdownIt("commonmark").enable("table").enable("strikethrough")
_TERMINATORS = ".!?。！？"
_CLOSERS = "\"'”’»)]}」』】）"
_BLOCK_BOUNDARY = "\u2029"


def sentence_count_issues(
    document: AnswerDocument, requirements: Sequence[SentenceCountRequirement], *, partial: bool = False,
) -> list[str]:
    required = {index: requirement for index, requirement in enumerate(requirements, 1)
                if requirement.mode == "required"}
    if not required:
        return []
    scoped = {index for index, requirement in enumerate(requirements, 1) if requirement.scope is not None}
    needs_scopes = any(requirement.scope is not None for requirement in required.values())
    counts = dict.fromkeys(scoped, 0)
    total = 0
    invalid_scope = False
    for group in _prose_groups(document):
        for owners in _sentence_owners(group):
            total += 1
            if needs_scopes and (len(owners) != 1 or not owners <= scoped):
                invalid_scope = True
            if len(owners) == 1:
                scope = next(iter(owners))
                if scope in counts:
                    counts[scope] += 1
    issues = ["sentence_scope"] if invalid_scope else []
    for index, requirement in required.items():
        count = total if requirement.scope is None else counts[index]
        absent_partial_scope = partial and requirement.scope is not None and count == 0
        if ((count < requirement.minimum and not absent_partial_scope)
                or (requirement.maximum is not None and count > requirement.maximum)):
            issues.append(f"sentence_count:{index}")
    return issues


def _prose_unit(unit: ContentUnit) -> tuple[str, int | None]:
    return content_unit_markdown(unit), unit.sentence_scope


def _prose_groups(document: AnswerDocument) -> Iterator[list[tuple[str, int | None]]]:
    for block in document.blocks:
        if block.type == "paragraph":
            yield [_prose_unit(unit) for unit in block.content]
        elif block.type == "list":
            for unit in block.items:
                yield [_prose_unit(unit)]
        elif block.type == "table":
            for unit in [*block.columns, *(unit for row in block.rows for unit in row)]:
                yield [_prose_unit(unit)]
        elif block.type == "heading":
            yield [_prose_unit(block.content)]
        elif block.inline:
            yield [(inline_code_markdown(block.content.text), block.content.sentence_scope)]


def _visible_prose(markdown: str) -> str:
    tokens = _MARKDOWN.parse(markdown)
    blocks = []
    for token in tokens:
        if token.type != "inline":
            continue
        parts = []
        for child in token.children or []:
            if child.type in {"text", "image"}:
                parts.append(child.content)
            elif child.type == "code_inline":
                parts.append("\ufffc")
            elif child.type in {"softbreak", "hardbreak"}:
                parts.append("\n")
        blocks.append("".join(parts))
    text = _BLOCK_BOUNDARY.join(blocks)
    # Ordinary paragraph fragments can continue across provenance units. Other
    # Markdown blocks retain their boundaries even when a unit ends mid-sentence.
    if [token.type for token in tokens] != ["paragraph_open", "inline", "paragraph_close"]:
        return _BLOCK_BOUNDARY + text + _BLOCK_BOUNDARY
    return text


def _sentence_owners(group: list[tuple[str, int | None]]) -> Iterator[set[int | None]]:
    text_parts = [_visible_prose(text) for text, _ in group]
    text = " ".join(text_parts)
    owners = []
    for index, (part, (_, scope)) in enumerate(zip(text_parts, group, strict=True)):
        if index:
            owners.append(None)
        owners.extend([scope] * len(part))
    sentence_scopes: set[int | None] = set()
    index = 0
    while index < len(text):
        char = text[index]
        if char == _BLOCK_BOUNDARY:
            if sentence_scopes:
                yield sentence_scopes
                sentence_scopes = set()
            index += 1
            continue
        if unicodedata.category(char)[0] not in {"P", "Z", "C"}:
            sentence_scopes.add(owners[index])
        if char not in _TERMINATORS:
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in _TERMINATORS + _CLOSERS:
            end += 1
        # Full stops inside decimals and unspaced identifiers are literal text.
        if char == "." and end < len(text) and not text[end].isspace():
            index += 1
            continue
        if sentence_scopes:
            yield sentence_scopes
            sentence_scopes = set()
        index = end
    if sentence_scopes:
        yield sentence_scopes
