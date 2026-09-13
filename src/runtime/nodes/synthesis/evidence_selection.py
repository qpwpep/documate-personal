from __future__ import annotations

import ast
import json
import re
import textwrap

from src.core.evidence import EvidenceRef, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.table_selection import table_excerpt, table_row_units

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


def matches_file_scope(item: EvidenceRef, task: RetrievalTask) -> bool:
    return not task.requirement.file_ids or upload_file_id(item) in task.requirement.file_ids


def requirement_coverage(
    task: RetrievalTask, evidence: list[EvidenceRef], requirement_ids: dict[str, list[str]],
) -> dict:
    """Describe literal coverage of the supplied ranges, independently of answer generation."""
    associated = [item for item in evidence
                  if task.requirement_id in requirement_ids.get(item.id, [])
                  and item.route == task.route and matches_file_scope(item, task)]
    missing = missing_literal_aspects(task.requirement.aspects, [item.excerpt for item in associated])
    coverage = {
        "evidence_ids": [item.id for item in associated],
        "present_aspects": [aspect for aspect in task.requirement.aspects if aspect not in missing],
        "missing_aspects": missing, "is_partial": not associated or bool(missing),
    }
    if task.requirement.file_ids:
        covered_files = sorted({upload_file_id(item) for item in associated})
        missing_files = [file_id for file_id in task.requirement.file_ids if file_id not in covered_files]
        missing_by_file = {
            file_id: missing_literal_aspects(task.requirement.aspects,
                                            [item.excerpt for item in associated if upload_file_id(item) == file_id])
            for file_id in task.requirement.file_ids
        }
        coverage.update(requested_file_ids=task.requirement.file_ids, covered_file_ids=covered_files,
                        missing_file_ids=missing_files, missing_aspects_by_file=missing_by_file,
                        is_partial=bool(missing_files) or any(missing_by_file.values()))
    return coverage


def missing_evidence_requirement_ids(
    planner_output: PlannerOutput, evidence: list[EvidenceRef], requirement_ids: dict[str, list[str]],
    *, strict: bool = False,
) -> list[str]:
    # Historical untagged route-only packets have no requirement-level contract.
    if not strict and not requirement_ids and not any(task.requirement.specified for task in planner_output.tasks):
        return []
    return [task.requirement_id for task in planner_output.tasks
            if requirement_coverage(task, evidence, requirement_ids)["is_partial"]]


def contains_evidence_range(outer: EvidenceRef, inner: EvidenceRef) -> bool:
    """Ranges may share a reference only within the same captured source element."""
    if outer.snapshot.snapshot_id != inner.snapshot.snapshot_id or outer.element != inner.element:
        return False
    if outer.selection.cell_ids or inner.selection.cell_ids:
        return bool(inner.selection.cell_ids) and set(inner.selection.cell_ids).issubset(outer.selection.cell_ids)
    outer_end = outer.selection.end if outer.selection.end is not None else len(outer.element.text)
    inner_end = inner.selection.end if inner.selection.end is not None else len(inner.element.text)
    return outer.selection.start <= inner.selection.start and inner_end <= outer_end


def tasks_for_hit(hit: SearchHit, planner_output: PlannerOutput) -> list[RetrievalTask]:
    """Keep the retrieval requirement attached to a hit; old untagged hits use their route."""
    return [
        task for task in planner_output.tasks
        if task.route == hit.evidence.route
        and (not hit.requirement_id or task.requirement_id == hit.requirement_id)
        and matches_file_scope(hit.evidence, task)
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
        if (hit.evidence.route == "upload" and any(task.requirement.file_ids for task in planner_output.tasks)
                and not tasks_for_hit(hit, planner_output)):
            continue
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
        for file_id in task.requirement.file_ids or [None]:
            scoped = [hit for hit in candidates if file_id is None or upload_file_id(hit.evidence) == file_id]
            match = min(scoped, key=lambda hit: rank_key(hit, task), default=None)
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


def _line_ranges(text: str) -> list[tuple[int, int]]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return list(zip(offsets, offsets[1:]))


def _code_ranges(item: EvidenceRef) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Use original line offsets for Python statements, falling back to complete source lines."""
    raw_lines = _line_ranges(item.excerpt)
    offsets = [0, *(end for _, end in raw_lines)]
    lines = [
        (start, end) for start, end in raw_lines
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


def _literal_code_anchors(text: str, terms: list[str]) -> list[tuple[int, int]]:
    anchors = set()
    for term in terms:
        pattern = r"\s+".join(re.escape(word) for word in term.split())
        anchors.update((match.start(), match.end()) for match in re.finditer(
            rf"(?<!\w){pattern}(?!\w)", text, re.IGNORECASE))
    return sorted(anchors)


def _bounded_code_range(
    text: str, span: tuple[int, int], *, limit: int, anchors: list[tuple[int, int]],
) -> tuple[int, int]:
    """Clip within a source window, retaining a whole literal anchor whenever it fits."""
    left, right = span
    width = min(limit, right - left)
    contained = [(start, end) for start, end in anchors if left <= start and end <= right]
    fitting = [(start, end) for start, end in contained if end - start <= width]
    if fitting:
        start, end = fitting[0]
        offset = max(left, end - width, min(start - width // 4, right - width))
    else:
        target = contained[0][0] if contained else left + next(
            (index for index, char in enumerate(text[left:right]) if not char.isspace()), 0)
        offset = max(left, min(target, right - width))
    return offset, offset + width


def select_table_evidence(
    item: EvidenceRef, *, limit: int, query: str = "", task: RetrievalTask | None = None,
) -> EvidenceRef | None:
    """Fit complete row/header units inside the retrieved cell selection."""
    if item.element.table is None or not item.selection.cell_ids or limit <= 0:
        return None
    if len(item.excerpt) <= limit:
        return item
    topics = _tokens(query) | (_task_tokens(task) if task is not None else set())
    topics.update(re.findall(r"\d+(?:[.,]\d+)*", query + (" " + task.query if task is not None else "")))
    aspects = task.requirement.aspects if task is not None else []
    units = table_row_units(item.element, allowed_cell_ids=item.selection.cell_ids)

    def rank(unit: list[str]) -> tuple[int, int, int]:
        text = table_excerpt(item.element.table, unit)
        terms = _tokens(text) | set(re.findall(r"\d+(?:[.,]\d+)*", text))
        return (len(aspects) - len(missing_literal_aspects(aspects, [text])), len(terms & topics), -len(text))

    selected_ids: list[str] = []
    for unit in sorted(units, key=rank, reverse=True):
        combined = list(dict.fromkeys([*selected_ids, *unit]))
        if len(table_excerpt(item.element.table, combined)) <= limit:
            selected_ids = combined
    if not selected_ids:
        return None
    return build_evidence(snapshot=item.snapshot, element=item.element, cell_ids=selected_ids)


def select_evidence_range(
    item: EvidenceRef, *, limit: int, query: str = "", task: RetrievalTask | None = None,
    expand_context: bool = True, allow_partial: bool = True, complete_char_limit: int | None = None,
) -> EvidenceRef:
    """Choose source ranges; reservation may defer whole code units to a larger allowance."""
    limit = max(0, min(limit, len(item.excerpt)))
    if len(item.excerpt) <= limit and expand_context:
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
            anchors = _literal_code_anchors(text, task.requirement.aspects if task is not None else [])
            if not anchors:
                # A query may name only the suffix of a long qualified identifier.
                anchors = _literal_code_anchors(text, sorted(focus))

            def related(span: tuple[int, int]) -> bool:
                return any(span[0] <= start and end <= span[1] for start, end in anchors)

            relevant_statements = [span for span in statements if related(span)]
            candidates = [span for span in relevant_statements if span[1] - span[0] <= limit]
            # Initial fair shares are not the final budget. Preserve a complete
            # statement's opportunity to use space left by smaller requirements.
            deferred = any(limit < end - start <= (complete_char_limit or 0)
                           for start, end in relevant_statements)
            if not candidates and not deferred:
                candidates = [span for span in units if span[1] - span[0] <= limit
                              and text[span[0]:span[1]].strip() and (not anchors or related(span))]
            if not candidates:
                raw_lines = _line_ranges(text)
                raw_ranges = list(raw_lines)
                # Literal coverage normalizes whitespace, so an anchor may span
                # lines. Keep those original line boundaries as one candidate.
                for start, end in anchors:
                    if not any(left <= start and end <= right for left, right in raw_lines):
                        raw_ranges.append((next(left for left, right in raw_lines if left <= start < right),
                                           next(right for left, right in raw_lines if left < end <= right)))
                raw_ranges = [span for span in raw_ranges if text[span[0]:span[1]].strip()
                              and (not anchors or related(span))]
                best = max(raw_ranges, key=relevance, default=(0, 0))
                deferred = deferred or (best in units and limit < best[1] - best[0] <= (complete_char_limit or 0))
                start, end = ((0, 0) if deferred or not allow_partial else
                              _bounded_code_range(text, best, limit=limit, anchors=anchors))
                return build_evidence(snapshot=item.snapshot, element=item.element,
                                      start=item.selection.start + start, end=item.selection.start + end)
            best = max(candidates, key=(relevance if expand_context else
                       lambda span: (*relevance(span)[:2], -(span[1] - span[0]), *relevance(span)[2:])))
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
                while expand_context:
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
