from __future__ import annotations

import ast
import json
import re
import textwrap

from src.core.evidence import EvidenceRef, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*|[가-힣]{2,}")
_STOPWORDS = {
    "uploaded", "upload", "file", "the", "this", "with", "from", "official", "docs", "code",
    "documentation", "reference", "parameter", "parameters", "option", "options", "explain",
    "a", "an", "to", "of", "in", "and", "or",
    "공식", "문서", "설명", "매개변수", "파라미터", "옵션",
}


def _tokens(text: str) -> set[str]:
    tokens = {token.lower().rstrip(".-") for token in _TOKEN_PATTERN.findall(text)}
    tokens.update(token.rsplit(".", 1)[-1] for token in tuple(tokens) if "." in token)
    return {token for token in tokens if token and token not in _STOPWORDS}


def missing_literal_aspects(aspects: list[str], excerpts: list[str]) -> list[str]:
    """Report missing literal anchors in model-visible text, without asserting semantic support."""
    normalized = [" ".join(excerpt.split()).casefold() for excerpt in excerpts]
    return [
        aspect for aspect in aspects
        if not any(
            re.search(rf"(?<!\w){re.escape(' '.join(aspect.split()).casefold())}(?!\w)", excerpt)
            for excerpt in normalized
        )
    ]


def route_for_evidence(item: EvidenceRef) -> str:
    return item.route


def upload_file_id(item: EvidenceRef) -> str:
    return str(item.element.metadata.get("file_id") or "") if item.route == "upload" else ""


def tasks_for_hit(hit: SearchHit, planner_output: PlannerOutput) -> list[RetrievalTask]:
    """Keep the retrieval requirement attached to a hit; old untagged hits use their route."""
    return [
        task for task in planner_output.tasks
        if task.route == hit.evidence.route
        and (not hit.requirement_id or task.requirement_id == hit.requirement_id)
    ]


def _task_tokens(task: RetrievalTask) -> set[str]:
    return _tokens(" ".join([
        task.query, *task.requirement.symbols, *task.requirement.aspects,
        task.requirement.library or "", task.requirement.version or "",
    ]))


def select_evidence_hits(
    *, user_input: str, hits: list[SearchHit], planner_output: PlannerOutput,
) -> list[SearchHit]:
    """Reserve a candidate for each requirement before adding further source ranges."""
    unique: dict[tuple[str, str], SearchHit] = {}
    for hit in hits:
        unique.setdefault((hit.evidence.id, hit.requirement_id), hit)

    def rank_key(hit: SearchHit, task: RetrievalTask | None = None) -> tuple[int, int]:
        evidence = hit.evidence
        text = " ".join([
            evidence.snapshot.title, evidence.excerpt,
            json.dumps(evidence.element.metadata, ensure_ascii=False),
        ])
        applicable = [task] if task is not None else tasks_for_hit(hit, planner_output)
        wanted = set().union(*(_task_tokens(item) for item in applicable)) if applicable else _tokens(user_input)
        matches = len(wanted.intersection(_tokens(text)))
        return (-matches, hit.rank)

    ranked = sorted(unique.values(), key=rank_key)
    first_per_requirement: list[SearchHit] = []
    selected_keys: set[tuple[str, str]] = set()
    for task in planner_output.tasks:
        candidates = [hit for hit in ranked if task in tasks_for_hit(hit, planner_output)]
        match = min(candidates, key=lambda hit: rank_key(hit, task), default=None)
        if match is not None:
            key = (match.evidence.id, match.requirement_id)
            if key not in selected_keys:
                first_per_requirement.append(match)
                selected_keys.add(key)
    return first_per_requirement + [
        hit for hit in ranked if (hit.evidence.id, hit.requirement_id) not in selected_keys
    ]


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    """Locate original paragraphs without splitting blank lines inside fenced code."""
    ranges: list[tuple[int, int]] = []
    start = 0
    offset = 0
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^[ \t]*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        if not line.strip() and fence is None:
            if text[start:offset].strip():
                ranges.append((start, offset))
            start = offset + len(line)
        offset += len(line)
    if text[start:].strip():
        ranges.append((start, len(text)))
    return ranges


def _sentence_ranges(text: str, start: int, end: int) -> list[tuple[int, int]]:
    boundaries = [start, *(start + match.end() for match in re.finditer(r"(?<=[.!?])\s+", text[start:end])), end]
    return [(left, right) for left, right in zip(boundaries, boundaries[1:]) if text[left:right].strip()]


def _code_ranges(item: EvidenceRef) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Use original line offsets for Python statements, falling back to complete source lines."""
    offsets = [0]
    for line in item.excerpt.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    lines = [
        (start, end) for start, end in zip(offsets, offsets[1:])
        if (item.selection.start + start == 0 or item.element.text[item.selection.start + start - 1] == "\n")
        and (
            item.selection.start + end == len(item.element.text)
            or item.element.text[item.selection.start + end - 1] == "\n"
            # AST end_col_offset excludes the newline, even for a complete final statement.
            or item.element.text[item.selection.start + end:item.selection.start + end + 1] in {"\r", "\n"}
        )
    ]
    if item.element.language not in {None, "python", "py"}:
        return [], lines
    try:
        tree = ast.parse(textwrap.dedent(item.excerpt))
    except (SyntaxError, ValueError):
        return [], lines
    starts = {start for start, _ in lines}
    ends = {end for _, end in lines}
    statements = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and node.end_lineno is not None:
            span = (offsets[node.lineno - 1], offsets[node.end_lineno])
            if span[0] in starts and span[1] in ends:
                statements.add(span)
    return sorted(statements), lines


def select_evidence_range(
    item: EvidenceRef, *, limit: int, query: str = "", task: RetrievalTask | None = None,
) -> EvidenceRef:
    """Choose relevant prose or code within the retrieved range, retaining exact source offsets."""
    if len(item.excerpt) <= limit:
        return item
    text = item.excerpt
    start, end = 0, limit
    if query or task is not None:
        topics = _tokens(query) | (_task_tokens(task) if task is not None else set())
        aspects = _tokens(" ".join(task.requirement.aspects)) if task is not None else set()
        focus = aspects or (topics - _tokens(item.snapshot.title)) or topics

        def relevance(span: tuple[int, int]) -> tuple[int, int, int, int, int]:
            passage = text[span[0]:span[1]]
            content = _tokens(passage)
            label = re.match(
                r"^[ \t]*(?:#{1,6}\s+([^\n]+)|\*\*([^*]+)\*\*|`([^`]+)`\s*:|([A-Za-z_][\w.-]*)\s*:)",
                passage,
            )
            label_terms = _tokens(next((value for value in label.groups() if value), "")) if label else set()
            focused_occurrences = sum(match.group().lower() in focus for match in _TOKEN_PATTERN.finditer(passage))
            return (
                len(label_terms & focus), len(content & focus), min(focused_occurrences, 8),
                len(content & topics), -span[0],
            )

        if item.element.kind == "code":
            statements, units = _code_ranges(item)
            fitting = [span for span in statements if span[1] - span[0] <= limit]
            relevant = [span for span in fitting if _tokens(text[span[0]:span[1]]) & focus]
            candidates = relevant or [span for span in units if span[1] - span[0] <= limit and text[span[0]:span[1]].strip()]
            if not candidates:
                return build_evidence(snapshot=item.snapshot, element=item.element, start=item.selection.start, end=item.selection.start)
            best = max(candidates, key=relevance)
        else:
            units = _paragraph_ranges(text)
            best = max(units, key=relevance) if units else (0, limit)
            if best[1] - best[0] > limit:
                units = _sentence_ranges(text, *best)
                best = max(units, key=relevance)
        if units:
            if best[1] - best[0] <= limit:
                start, end = best
                # Expand only across whole neighboring units. The chosen parameter
                # or sentence remains intact even when unrelated context is long.
                left = next(index for index, span in enumerate(units) if span[0] == start)
                right = next(index for index, span in enumerate(units) if span[1] == end)
                while True:
                    choices = []
                    if left > 0 and end - units[left - 1][0] <= limit:
                        choices.append((units[left - 1], "left"))
                    if right + 1 < len(units) and units[right + 1][1] - start <= limit:
                        choices.append((units[right + 1], "right"))
                    if not choices:
                        break
                    span, direction = max(choices, key=lambda choice: relevance(choice[0]))
                    if direction == "left":
                        left -= 1
                        start = span[0]
                    else:
                        right += 1
                        end = span[1]
            else:
                # An individual sentence can exceed the bound. Keep a bounded
                # original slice around the matched term; the packet marks it partial.
                matches = [
                    match.start() for match in _TOKEN_PATTERN.finditer(text, best[0], best[1])
                    if match.group().lower() in focus
                ]
                target = min(matches, default=best[0])
                start = max(best[0], min(target - limit // 4, best[1] - limit))
                end = start + limit
    return build_evidence(
        snapshot=item.snapshot, element=item.element,
        start=item.selection.start + start, end=item.selection.start + end,
    )
