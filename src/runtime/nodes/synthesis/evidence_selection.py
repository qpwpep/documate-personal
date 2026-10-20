from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
import json
import re
import textwrap

from src.core.evidence import EvidenceRef, SearchHit, build_evidence
from src.core.literal_matching import literal_aspect_present, missing_literal_aspects
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.python_source import PythonSourceMap
from src.core.table_selection import table_excerpt, table_row_units
from src.infra.tools.docs_search.sections import parse_doc_sections, scan_declarations
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget, literal_passage_anchors, upload_passage_targets

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


def route_for_evidence(item: EvidenceRef) -> str:
    return item.route


def upload_file_id(item: EvidenceRef) -> str:
    return str(item.element.metadata.get("file_id") or "") if item.route == "upload" else ""


def matches_file_scope(item: EvidenceRef, task: RetrievalTask) -> bool:
    return not task.requirement.file_ids or upload_file_id(item) in task.requirement.file_ids


def _missing_selected_definitions(symbols: list[str], evidence: list[EvidenceRef]) -> list[str]:
    """Check complete Python definition offsets when the original is parseable.

    Parsing an unknown language or failing to parse source cannot establish a
    semantic absence. Decisions handles those cases using its source context.
    """
    complete: set[str] = set()
    analyzed = False
    for item in evidence:
        if item.element.kind != "code" or item.element.language not in {None, "python", "py"}:
            continue
        try:
            tree = ast.parse(item.element.text)
        except (SyntaxError, ValueError, RecursionError):
            continue
        analyzed = True
        source_map = PythonSourceMap(item.element.text)
        selected_end = item.selection.end if item.selection.end is not None else len(item.element.text)
        pending = [(tree, "")]
        while pending:
            node, parent = pending.pop()
            child_parent = parent
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child_parent = f"{parent}.{node.name}" if parent else node.name
                start_line, start_column = source_map.definition_start(node)
                start = source_map.position(start_line, start_column)
                end = source_map.position(node.end_lineno, node.end_col_offset)
                if item.selection.start <= start and end <= selected_end:
                    complete.update(symbol for symbol in symbols if (
                        child_parent == symbol or child_parent.endswith("." + symbol)))
            pending.extend((child, child_parent) for child in ast.iter_child_nodes(node))
    return [symbol for symbol in symbols if symbol not in complete] if analyzed else []


def requirement_coverage(
    task: RetrievalTask, evidence: list[EvidenceRef], requirement_ids: dict[str, list[str]],
) -> dict:
    """Describe coverage of the supplied ranges, independently of answer generation."""
    associated = [item for item in evidence
                  if task.requirement_id in requirement_ids.get(item.id, [])
                  and item.route == task.route and matches_file_scope(item, task)]
    coverage = {"evidence_ids": [item.id for item in associated]}
    topic_symbols = task.requirement.symbols if task.route == "upload" and task.requirement.match == "topic" else []
    if task.route == "docs":
        # This is association coverage only. Decisions owns meaning and sufficiency.
        coverage["is_partial"] = not associated
        return coverage
    else:
        missing = missing_literal_aspects(task.requirement.aspects, associated)
        coverage.update(
            present_aspects=[aspect for aspect in task.requirement.aspects if aspect not in missing],
            missing_aspects=missing, is_partial=not associated or bool(missing),
        )
    if topic_symbols:
        missing_symbols = missing_literal_aspects(topic_symbols, associated)
        coverage.update(present_symbols=[symbol for symbol in topic_symbols if symbol not in missing_symbols],
                        missing_symbols=missing_symbols, is_partial=coverage["is_partial"] or bool(missing_symbols))
    if task.requirement.file_ids:
        covered_files = sorted({upload_file_id(item) for item in associated})
        missing_files = [file_id for file_id in task.requirement.file_ids if file_id not in covered_files]
        missing_by_file = {
            file_id: missing_literal_aspects(task.requirement.aspects,
                                            [item for item in associated if upload_file_id(item) == file_id])
            for file_id in task.requirement.file_ids
        }
        coverage.update(requested_file_ids=task.requirement.file_ids, covered_file_ids=covered_files,
                        missing_file_ids=missing_files, missing_aspects_by_file=missing_by_file,
                        is_partial=coverage["is_partial"] or bool(missing_files) or any(missing_by_file.values()))
        if topic_symbols:
            symbols_by_file = {file_id: missing_literal_aspects(topic_symbols,
                [item for item in associated if upload_file_id(item) == file_id]) for file_id in task.requirement.file_ids}
            coverage.update(missing_symbols_by_file=symbols_by_file,
                            is_partial=coverage["is_partial"] or any(symbols_by_file.values()))
            missing_requirements = []
            for file_id in task.requirement.file_ids:
                if file_id in missing_files:
                    missing_requirements.append(f"file:{file_id}")
                    continue
                missing_requirements.extend(f"file:{file_id}:symbol:{symbol}" for symbol in symbols_by_file[file_id])
                missing_requirements.extend(f"file:{file_id}:aspect:{aspect}" for aspect in missing_by_file[file_id])
            coverage["missing_requirements"] = missing_requirements
    elif topic_symbols:
        coverage["missing_requirements"] = [*(f"symbol:{symbol}" for symbol in missing_symbols),
                                            *(f"aspect:{aspect}" for aspect in missing)]
    if task.route == "upload" and task.requirement.match == "definition":
        missing_definitions = _missing_selected_definitions(task.requirement.symbols, associated)
        by_file = {file_id: _missing_selected_definitions(task.requirement.symbols, [
            item for item in associated if upload_file_id(item) == file_id])
            for file_id in task.requirement.file_ids}
        coverage.update(missing_definitions=missing_definitions, missing_definitions_by_file=by_file,
                        is_partial=coverage["is_partial"] or bool(missing_definitions) or any(by_file.values()))
    return coverage


def missing_evidence_requirement_ids(
    planner_output: PlannerOutput, evidence: list[EvidenceRef], requirement_ids: dict[str, list[str]],
) -> list[str]:
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
    """Associate only the requirement assigned by the retrieval executor."""
    return [
        task for task in planner_output.tasks
        if task.route == hit.evidence.route
        and task.requirement_id == hit.requirement_id
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
            rf"(?<!\w){pattern}(?!\w)", text))
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
    aspects = literal_passage_anchors(task) if task is not None else []
    units = table_row_units(item.element, allowed_cell_ids=item.selection.cell_ids)

    def rank(unit: list[str]) -> tuple[int, int, int]:
        text = table_excerpt(item.element.table, unit)
        terms = _tokens(text) | set(re.findall(r"\d+(?:[.,]\d+)*", text))
        return (sum(literal_aspect_present(aspect, text, case_sensitive=False) for aspect in aspects), len(terms & topics), -len(text))

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
        aspects = _tokens(" ".join(literal_passage_anchors(task))) if task is not None else set()
        focus = aspects or (topics - _tokens(item.snapshot.title)) or topics

        def relevance(span: tuple[int, int]) -> tuple[int, ...]:
            passage = text[span[0]:span[1]]
            content = _tokens(passage)
            label = re.match(
                r"^[ \t]*(?:#{1,6}\s+([^\n]+)|\*\*([^*]+)\*\*|`([^`]+)`\s*:|([A-Za-z_][\w.-]*)\s*:)",
                passage,
            )
            label_terms = _tokens(next((value for value in label.groups() if value), "")) if label else set()
            focused_occurrences = sum(match.group().lower() in focus for match in _TOKEN_PATTERN.finditer(passage))
            lexical = (
                len(label_terms & focus), len(content & focus), min(focused_occurrences, 8),
                len(content & topics), -span[0],
            )
            return lexical

        if item.element.kind == "code":
            statements, units = _code_ranges(item)
            anchors = _literal_code_anchors(text, literal_passage_anchors(task) if task is not None else [])
            if not anchors:
                # A query may name only the suffix of a long qualified identifier.
                raw_terms = _TOKEN_PATTERN.findall(" ".join([
                    query, task.query if task is not None else "",
                    *(task.requirement.symbols if task is not None else []),
                    *(task.requirement.aspects if task is not None else []),
                ]))
                terms = {term.rstrip(".-") for term in raw_terms}
                terms.update(term.rsplit(".", 1)[-1] for term in tuple(terms) if "." in term)
                anchors = _literal_code_anchors(text, sorted(term for term in terms if term.lower() in focus))

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
                if not allow_partial or best[1] - best[0] <= (complete_char_limit or 0):
                    return build_evidence(snapshot=item.snapshot, element=item.element,
                                          start=item.selection.start, end=item.selection.start)
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


def _doc_passages(item: EvidenceRef, limit: int) -> list[EvidenceRef]:
    """Propose original contiguous ranges without certifying their meaning."""
    if limit <= 0:
        return []
    text = item.excerpt
    units = []
    for start, end in _paragraph_ranges(text):
        if end - start <= limit:
            units.append((start, end))
        else:
            units.extend((start + left, start + right) for left, right in _line_ranges(text[start:end])
                         if right - left <= limit and text[start + left:start + right].strip())
            units.extend((left, right) for left, right in _sentence_ranges(text, start, end)
                         if right - left <= limit)
    units = sorted(set(units))
    spans = set(units)
    # Parameter labels and descriptions may be separated by a blank line.
    spans.update((left[0], right[1]) for left, right in zip(units, units[1:])
                 if right[1] - left[0] <= limit)
    # A parameter's value/condition may occur several lines after its label.
    # Keep short candidates for allocation, and add the longest fitting run
    # from each unit. A monotone end cursor adds only O(n) candidates instead
    # of enumerating every pair of boundaries in a densely wrapped source.
    right = 0
    for left, (start, _) in enumerate(units):
        right = max(left, right)
        while right + 1 < len(units) and units[right + 1][1] - start <= limit:
            right += 1
        spans.add((start, units[right][1]))
    # Declaration parsing supplies candidate boundaries, never an acceptance gate.
    spans.update((declaration.start, declaration.end)
                 for declaration in scan_declarations(text)
                 if declaration.end - declaration.start <= limit)
    if text.strip() and len(text) <= limit:
        spans.add((0, len(text)))
    return [build_evidence(snapshot=item.snapshot, element=item.element,
                           start=item.selection.start + start, end=item.selection.start + end)
            for start, end in sorted(spans)]


_DOC_CANDIDATES_PER_SOURCE = 8
_DOC_CANDIDATES_TOTAL = 96


def prepare_evidence_candidates(
    *, user_input: str, hits: list[SearchHit], planner_output: PlannerOutput,
    budget: RetrievedEvidenceBudget,
    include_candidate: Callable[[SearchHit], bool] | None = None,
) -> list[SearchHit]:
    """Prepare finite, budget-sized source selections before semantic assessment.

    Paragraphs, lines and declarations provide boundaries. Lexical matches only
    order alternatives; unrecognized prose always retains a default candidate.
    Round-robin allocation keeps one source from consuming the candidate pool.
    The caller can exclude exact ranges already rejected for the current
    judgment input before they consume any source, requirement or total cap.
    """
    pools: list[list[SearchHit]] = []

    def add_pool(candidates: Iterable[SearchHit]) -> None:
        pool = []
        for candidate in candidates:
            if include_candidate is not None and not include_candidate(candidate):
                continue
            pool.append(candidate)
            if len(pool) == _DOC_CANDIDATES_PER_SOURCE:
                break
        pools.append(pool)

    limits = {task.requirement_id: min(32, max(12, 2 * (
        len(task.requirement.aspects) + len(task.requirement.evidence_needs) + 1)))
        for task in planner_output.tasks}
    for hit in select_evidence_hits(user_input=user_input, hits=hits, planner_output=planner_output):
        item = hit.evidence
        applicable = tasks_for_hit(hit, planner_output)
        task = applicable[0] if applicable else None
        if item.route != "docs":
            # Upload definitions and table cells keep their existing selector.
            add_pool([hit])
            continue
        limit = min(budget.max_excerpt_chars, budget.max_total_excerpt_chars)
        if limit <= 0 or not item.excerpt.strip():
            continue
        if len(item.excerpt) <= limit:
            add_pool([hit])
            continue
        if item.element.kind == "table":
            selected = select_table_evidence(item, limit=limit, query=user_input, task=task)
            add_pool([hit.model_copy(update={"evidence": selected})] if selected else [])
            continue
        proposals = _doc_passages(item, limit)
        fallback = select_evidence_range(item, limit=limit, query=user_input, task=task)
        proposals.append(fallback)
        topics = _task_tokens(task) if task else _tokens(user_input)
        symbols = task.requirement.symbols if task else []
        identity_terms = _tokens(" ".join([*symbols, task.requirement.library or ""])) if task else set()
        identity_terms.update(part.lower() for symbol in symbols for part in symbol.split("."))
        focus_terms = _tokens(" ".join(task.requirement.aspects)) if task else set()
        sections = parse_doc_sections(item.excerpt, title=item.snapshot.title, url=item.snapshot.source_uri)
        owned_ranges = [(section.start, section.end) for section in sections
                        if any(section.owner == symbol or section.owner.endswith("." + symbol)
                               for symbol in symbols)]
        owner_groups: list[tuple[int, int]] = []
        for start, end in owned_ranges:
            if owner_groups and owner_groups[-1][1] == start:
                owner_groups[-1] = (owner_groups[-1][0], end)
            else:
                owner_groups.append((start, end))
        # One API section can switch from signature to example and back to
        # prose. Offer its complete contiguous context before ranking; otherwise
        # only a short prefix may fit without crossing into the following API.
        for start, end in owner_groups:
            scoped = build_evidence(snapshot=item.snapshot, element=item.element,
                start=item.selection.start + start, end=item.selection.start + end)
            proposals.extend(_doc_passages(scoped, limit))

        def rank(candidate: EvidenceRef) -> tuple[float, int, int, int, int, int]:
            content = _tokens(candidate.excerpt)
            left = candidate.selection.start - item.selection.start
            right = left + len(candidate.excerpt)
            owned_chars = sum(max(0, min(right, end) - max(left, start)) for start, end in owned_ranges)
            # An adjacent API's cross-links repeat the requested qualified name
            # without supplying its section. Structure is only a ranking hint:
            # missing/ambiguous ownership never rejects a candidate.
            return (owned_chars / max(1, len(candidate.excerpt)), len(content & focus_terms),
                    len(content & (topics - identity_terms)), len(content & topics),
                    len(candidate.excerpt), -candidate.selection.start)

        ranked = sorted(proposals, key=rank, reverse=True)
        # Round-robin caps can leave only one candidate from each source. Give
        # that slot the best contextual range, never the shortest anchor alone.
        chosen: dict[str, EvidenceRef] = {ranked[0].id: ranked[0]}
        # Each explicit anchor gets a complete fitting neighborhood opportunity.
        # An absent anchor does not reject the default or any other candidate.
        for anchor in task.requirement.aspects if task else []:
            anchored = [candidate for candidate in ranked
                        if literal_aspect_present(anchor, candidate.excerpt, case_sensitive=False)]
            if anchored:
                candidate = min(anchored, key=lambda value: (len(value.excerpt), value.selection.start))
                chosen[candidate.id] = candidate
                neighborhood = max(
                    (value for value in anchored if value.selection.start == candidate.selection.start),
                    key=lambda value: len(value.excerpt),
                )
                chosen[neighborhood.id] = neighborhood
        chosen.setdefault(fallback.id, fallback)
        # A dense leading section must not hide the document's trailing units
        # or its complete declarations merely because they share fewer tokens.
        for declaration in scan_declarations(item.excerpt):
            if declaration.end - declaration.start <= limit:
                candidate = build_evidence(snapshot=item.snapshot, element=item.element,
                    start=item.selection.start + declaration.start, end=item.selection.start + declaration.end)
                chosen.setdefault(candidate.id, candidate)
        if proposals:
            trailing = max(proposals, key=lambda value: (value.selection.end, -len(value.excerpt)))
            chosen.setdefault(trailing.id, trailing)
        for candidate in ranked:
            if not any(contains_evidence_range(selected, candidate)
                       or contains_evidence_range(candidate, selected) for selected in chosen.values()):
                chosen.setdefault(candidate.id, candidate)
        for candidate in ranked:
            chosen.setdefault(candidate.id, candidate)
        add_pool(hit.model_copy(update={"evidence": candidate}) for candidate in chosen.values())
    result: list[SearchHit] = []
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    docs_count = 0
    for index in range(max(map(len, pools), default=0)):
        for pool in pools:
            if index >= len(pool):
                continue
            candidate = pool[index]
            key = (candidate.evidence.id, candidate.requirement_id)
            if key in seen:
                continue
            if candidate.evidence.route == "docs":
                if (docs_count >= _DOC_CANDIDATES_TOTAL
                        or counts.get(candidate.requirement_id, 0) >= limits.get(candidate.requirement_id, 12)):
                    continue
                docs_count += 1
                counts[candidate.requirement_id] = counts.get(candidate.requirement_id, 0) + 1
            seen.add(key)
            result.append(candidate)
    return result


def select_evidence_packet(
    evidence: list[EvidenceRef], *, budget: RetrievedEvidenceBudget, query: str = "",
    requirements_by_evidence: dict[str, list[RetrievalTask]] | None = None,
    semantic_support: dict[str, dict[str, set[str]]] | None = None,
) -> tuple[list[EvidenceRef], dict[str, list[str]]]:
    """Reserve required passages before expanding context, charging shared ranges only once."""
    packet: list[EvidenceRef] = []
    requirement_ids: dict[str, list[str]] = {}
    remaining = budget.max_total_excerpt_chars
    candidates: dict[tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], tuple[EvidenceRef, RetrievalTask | None]] = {}
    contracts: dict[str, RetrievalTask] = {}
    support = semantic_support or {}
    selected_labels: dict[str, set[str]] = {}
    for item in evidence:
        tasks = (requirements_by_evidence or {}).get(item.id) or [None]
        for task in tasks:
            if task is not None:
                contracts[task.requirement_id] = task
            targets = (upload_passage_targets(task) if task is not None and task.route != "docs"
                       else [(None, None)])
            for file_id, aspect in targets:
                if file_id is not None and upload_file_id(item) != file_id:
                    continue
                changes = {}
                if (task is not None and task.route == "upload" and task.requirement.match == "topic"
                        and len(literal_passage_anchors(task)) > 1):
                    changes["symbols"] = [aspect] if aspect in task.requirement.symbols else []
                    changes["aspects"] = [aspect] if aspect in task.requirement.aspects else []
                elif task is not None and task.route != "docs" and len(task.requirement.aspects) > 1:
                    changes["aspects"] = [aspect]
                focused = (
                    task.model_copy(update={"requirement": task.requirement.model_copy(update=changes)})
                    if changes else task
                )
                key = (
                    item.id, focused.requirement_id if focused else "",
                    tuple(focused.requirement.symbols) if focused else (),
                    tuple(focused.requirement.aspects) if focused else (),
                    tuple(focused.requirement.evidence_needs) if focused else (),
                )
                candidates.setdefault(key, (item, focused))

    def group(candidate: tuple[EvidenceRef, RetrievalTask | None]) -> str:
        item, task = candidate
        if task is not None and task.requirement.file_ids:
            return f"{task.requirement_id}:file:{upload_file_id(item)}"
        return task.requirement_id if task else f"route:{item.route}"

    first: dict[str, tuple[EvidenceRef, RetrievalTask | None]] = {}
    rest = []
    for candidate in candidates.values():
        key = group(candidate)
        if key not in first:
            first[key] = candidate
        else:
            rest.append(candidate)
    ordered = [*first.values(), *rest]
    origins: dict[str, tuple[EvidenceRef, RetrievalTask | None]] = {}

    def associate(selected: EvidenceRef, task: RetrievalTask | None) -> None:
        if task is not None:
            associated = requirement_ids.setdefault(selected.id, [])
            if task.requirement_id not in associated:
                associated.append(task.requirement_id)

    def needs_passage(item: EvidenceRef, task: RetrievalTask | None) -> bool:
        if task is not None and task.route == "docs":
            wanted = {"relevance", *(f"aspect:{aspect}" for aspect in task.requirement.aspects),
                      *(f"need:{need}" for need in task.requirement.evidence_needs)}
            return bool(wanted - selected_labels.get(task.requirement_id, set()))
        selected_ranges = [selected for selected in packet
                    if (task is None and selected.route == item.route) or (
                        task is not None and task.requirement_id in requirement_ids.get(selected.id, [])
                        and (not task.requirement.file_ids or upload_file_id(selected) == upload_file_id(item)))]
        return not selected_ranges or bool(task and missing_literal_aspects(literal_passage_anchors(task), selected_ranges))

    # All associations are attached before testing capacity: sharing a cropped
    # range is free even when the item or character budget has been exhausted.
    def share_existing(item: EvidenceRef, task: RetrievalTask | None) -> None:
        for selected in packet:
            if item.route == "docs":
                if selected.id == item.id and (semantic_support is None or task is None
                        or support.get(item.id, {}).get(task.requirement_id)):
                    associate(selected, task)
                continue
            if contains_evidence_range(item, selected):
                associate(selected, task)

    def replaced_ranges(selected: EvidenceRef) -> list[EvidenceRef]:
        if selected.route == "docs":
            return []
        return [old for old in packet if contains_evidence_range(selected, old)
                and all(any(candidate_task is not None and candidate_task.requirement_id == requirement_id
                            and contains_evidence_range(candidate, selected)
                            for candidate, candidate_task in ordered)
                        for requirement_id in requirement_ids.get(old.id, []))]

    def admit(selected: EvidenceRef, item: EvidenceRef, task: RetrievalTask | None) -> bool:
        nonlocal remaining
        for old in packet:
            if selected.route == "docs" and old.id != selected.id:
                continue
            if contains_evidence_range(old, selected) and contains_evidence_range(item, old):
                associate(old, task)
                return False
        replaced = replaced_ranges(selected)
        cost = len(selected.excerpt) - sum(len(old.excerpt) for old in replaced)
        if cost > remaining or len(packet) - len(replaced) >= budget.max_items:
            return False
        associations = {requirement_id for old in replaced for requirement_id in requirement_ids.get(old.id, [])}
        for old in replaced:
            packet.remove(old)
            origins.pop(old.id, None)
            requirement_ids.pop(old.id, None)
        packet.append(selected)
        origins[selected.id] = (item, task)
        requirement_ids[selected.id] = sorted(associations)
        associate(selected, task)
        remaining -= cost
        return True

    # Docs ranges were prepared before Decisions. Normal selection keeps their
    # exact identities; compact selection creates new identities for rechecking.
    doc_proposals: dict[tuple[str, str], tuple[EvidenceRef, EvidenceRef, RetrievalTask]] = {}
    seen_docs = set()
    for item, focused in ordered:
        if focused is None or focused.route != "docs" or (item.id, focused.requirement_id) in seen_docs:
            continue
        seen_docs.add((item.id, focused.requirement_id))
        task = contracts[focused.requirement_id]
        labels = support.get(item.id, {}).get(task.requirement_id, set())
        if semantic_support is not None and not labels:
            continue
        if len(item.excerpt) <= budget.max_excerpt_chars:
            proposals = [item]
        elif item.element.kind == "table":
            selected = select_table_evidence(item, limit=budget.max_excerpt_chars, query=query, task=task)
            proposals = [selected] if selected is not None else []
        else:
            proposals = [select_evidence_range(item, limit=budget.max_excerpt_chars, query=query, task=task)]
        for selected in proposals:
            doc_proposals.setdefault((selected.id, task.requirement_id), (selected, item, task))

    def reserve_docs(allowances: dict[str, int] | None = None) -> None:
        spent: dict[str, int] = {}
        while True:
            best = None
            best_score = None
            for selected, original, task in doc_proposals.values():
                key = task.requirement_id
                if key in requirement_ids.get(selected.id, []):
                    continue
                existing = any(item.id == selected.id for item in packet)
                cost = 0 if existing else len(selected.excerpt)
                if (cost > remaining or (not existing and len(packet) >= budget.max_items)
                        or (allowances is not None and cost > allowances[key] - spent.get(key, 0))):
                    continue
                labels = support.get(original.id, {}).get(key, {"relevance"} if semantic_support is None else set())
                gain = labels - selected_labels.get(key, set())
                if allowances is not None and any(
                    other_task.requirement_id == key
                    and labels < support.get(other_original.id, {}).get(key, set())
                    and allowances[key] - spent.get(key, 0) < len(other.excerpt) <= remaining
                    for other, other_original, other_task in doc_proposals.values()
                ):
                    # Defer the larger, better-supported range until small
                    # requirements have released their unused fair shares.
                    continue
                if not gain and allowances is not None:
                    continue
                if not gain and any(
                    contains_evidence_range(existing, selected) or contains_evidence_range(selected, existing)
                    for existing in packet if key in requirement_ids.get(existing.id, [])
                ):
                    continue
                # A label is an allocation hint, not a proof that this packet
                # satisfies the task. Exact selected ranges are judged afterward.
                capacity = remaining if allowances is None else min(
                    remaining, allowances[key] - spent.get(key, 0))
                pressure = max(
                    max(0, cost) / max(1, capacity),
                    int(not existing) / max(1, budget.max_items - len(packet)),
                    1 / max(1, budget.max_total_excerpt_chars),
                )
                score = (not selected_labels.get(key), bool(gain), len(gain), len(gain) / pressure, -cost)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (selected, original, task, cost, labels)
            if best is None:
                return
            selected, original, task, cost, labels = best
            admit(selected, original, task)
            associate(selected, task)
            selected_labels.setdefault(task.requirement_id, set()).update(labels)
            spent[task.requirement_id] = spent.get(task.requirement_id, 0) + cost

    def add_candidate(
        item: EvidenceRef, task: RetrievalTask | None, allowance: int, *, require_anchor: bool = True,
        allow_partial: bool = True, complete_char_limit: int | None = None,
    ) -> None:
        if len(packet) >= budget.max_items or allowance <= 0:
            return
        if item.element.kind == "table":
            selected = select_table_evidence(
                item, limit=min(budget.max_excerpt_chars, allowance), query=query, task=task,
            )
            if selected is None:
                return
        else:
            limit = min(budget.max_excerpt_chars, allowance)
            if not limit:
                return
            selected = select_evidence_range(
                item, limit=limit, query=query, task=task, expand_context=False,
                allow_partial=allow_partial, complete_char_limit=complete_char_limit,
            )
        if not selected.excerpt.strip():
            return
        if (require_anchor and task is not None and task.route != "docs"
                and missing_literal_aspects(literal_passage_anchors(task), [selected])):
            return
        admit(selected, item, task)

    def fair_allowance() -> int:
        # Deferred earlier requirements still own a share of the remaining budget.
        pending = {(group(candidate), tuple(candidate[1].requirement.symbols) if candidate[1] else (),
                    tuple(candidate[1].requirement.aspects) if candidate[1] else (),
                    tuple(candidate[1].requirement.evidence_needs) if candidate[1] else ())
                   for candidate in ordered if needs_passage(*candidate)}
        return remaining // max(1, min(len(pending), budget.max_items - len(packet)))

    groups = {group(candidate) for candidate in ordered}
    reserve_docs({key: remaining // max(1, len(groups))
                  for key, task in contracts.items() if task.route == "docs"})

    for item, task in ordered:
        share_existing(item, task)
        if task is not None and task.route == "docs":
            continue
        if not needs_passage(item, task):
            continue
        add_candidate(item, task, fair_allowance(), complete_char_limit=budget.max_excerpt_chars)

    # Whole statements and table selections need unequal space. Revisit unmet
    # requirements using capacity left by smaller passages before optional text.
    while True:
        before = (len(packet), sum(map(len, requirement_ids.values())))
        for item, task in ordered:
            share_existing(item, task)
            if task is not None and task.route == "docs":
                continue
            if needs_passage(item, task):
                add_candidate(item, task, fair_allowance(), allow_partial=False)
        if before == (len(packet), sum(map(len, requirement_ids.values()))):
            break

    # If whole units cannot share the available budget, reserve partial source
    # ranges for the unmet requirements before adding optional sources/context.
    for item, task in ordered:
        share_existing(item, task)
        if task is not None and task.route == "docs":
            continue
        if needs_passage(item, task):
            add_candidate(item, task, fair_allowance())

    # Other routes have now reserved their passages. Release unused fair shares
    # to remaining docs obligations before spending anything on extra context.
    reserve_docs()

    # Additional sources can describe requested behavior beyond what is needed
    # to satisfy coverage. Select their relevant passages before introductory
    # context marks a source represented or spends its available space.
    for item, task in ordered:
        share_existing(item, task)
        if item.route == "docs":
            continue
        if any(original.id == item.id for original, _ in origins.values()):
            continue
        add_candidate(item, task, remaining, require_anchor=False)

    reserved = list(packet)
    for index, selected in enumerate(reserved):
        # Expanding an earlier passage may already have subsumed this range.
        if selected.id not in origins:
            continue
        original, task = origins[selected.id]
        if original.route == "docs" or original.element.kind == "table":
            continue
        limit = min(budget.max_excerpt_chars, len(selected.excerpt) + remaining // (len(reserved) - index))
        expanded = select_evidence_range(
            original, limit=limit, query=query, task=task,
        )
        if not contains_evidence_range(expanded, selected):
            continue
        if any(not any(candidate_task is not None and candidate_task.requirement_id == requirement_id
                       and contains_evidence_range(candidate, expanded)
                       for candidate, candidate_task in ordered)
               for requirement_id in requirement_ids.get(selected.id, [])):
            continue
        admit(expanded, original, task)
    packet = list({selected.id: selected for selected in packet}.values())
    requirement_ids = {key: value for key, value in requirement_ids.items()
                       if any(item.id == key for item in packet)}
    for item, task in ordered:
        share_existing(item, task)
    return packet, requirement_ids
