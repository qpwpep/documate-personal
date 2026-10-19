from __future__ import annotations

import ast
from collections.abc import Callable
import json
import re
import textwrap

from src.core.evidence import EvidenceRef, SearchHit, build_evidence
from src.core.literal_matching import literal_aspect_present, missing_literal_aspects
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.table_selection import table_excerpt, table_row_units
from src.infra.tools.docs_search.ranking import DocsAssessment, DocsAssessmentCache, assess_docs_evidence
from src.infra.tools.docs_search.requirements import resolve_requirement
from src.runtime.nodes.synthesis.budgets import RetrievedEvidenceBudget, literal_passage_anchors, requirement_passage_targets

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


def _assess_docs_ranges(task: RetrievalTask, evidence: list[EvidenceRef]) -> DocsAssessment:
    return assess_docs_evidence(task.query, evidence, task.requirement)


def _has_missing_label(missing: list[str], kind: str, value: str) -> bool:
    label = f"{kind}:{value}"
    return any(item == label or item.startswith(label + ":") for item in missing)


def _focused_docs_missing(assessment: DocsAssessment, task: RetrievalTask) -> list[str]:
    """A passage focus changes priority, never the contract used to establish support."""
    return [label for label in assessment.missing
            if (not label.startswith("aspect:") or any(_has_missing_label([label], "aspect", aspect)
                                                       for aspect in task.requirement.aspects))
            and (not label.startswith("need:") or any(_has_missing_label([label], "need", need)
                                                       for need in task.requirement.evidence_needs))]


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
        assessment = _assess_docs_ranges(task, associated)
        missing = [aspect for aspect in task.requirement.aspects if _has_missing_label(assessment.missing, "aspect", aspect)]
        missing_needs = [need for need in task.requirement.evidence_needs if _has_missing_label(assessment.missing, "need", need)]
        coverage.update(
            present_aspects=[aspect for aspect in task.requirement.aspects if aspect not in missing],
            missing_aspects=missing,
            present_needs=[need for need in task.requirement.evidence_needs if need not in missing_needs],
            missing_needs=missing_needs, missing_requirements=assessment.missing,
            is_partial=assessment.answerability != "covered",
        )
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
    coverage_task: RetrievalTask | None = None,
    covered_evidence: list[EvidenceRef] | None = None,
    docs_assessor: Callable[[RetrievalTask, list[EvidenceRef]], DocsAssessment] | None = None,
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

        docs_scores: dict[tuple[int, int], tuple[int, int, int]] = {}

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
            if task is None or task.route != "docs":
                return lexical
            if span not in docs_scores:
                selected = build_evidence(
                    snapshot=item.snapshot, element=item.element,
                    start=item.selection.start + span[0], end=item.selection.start + span[1],
                )
                # Score the contribution to this requirement's existing packet.
                # A new example need not repeat a parameter aspect already supplied.
                assessment = (docs_assessor or _assess_docs_ranges)(
                    coverage_task or task, [*(covered_evidence or []), selected])
                missing = _focused_docs_missing(assessment, task)
                docs_scores[span] = (not missing, -len(missing), -len(assessment.missing))
            return (*docs_scores[span], *lexical)

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


def _complete_doc_passages(item: EvidenceRef, limit: int, *, cache: DocsAssessmentCache) -> list[EvidenceRef]:
    """Offer complete source units, including a label and its adjacent description.

    A sentence cut out of a longer physical line cannot establish parameter
    coverage. Keep original paragraph/line boundaries and let the same strict
    coverage evaluator decide which of these ranges supplies a requirement.
    """
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
    spans = set(units)
    # Parameter labels and descriptions may be separated by a blank line.
    spans.update((left[0], right[1]) for left, right in zip(units, units[1:])
                 if right[1] - left[0] <= limit)
    # Paragraphs can join a multiline declaration to an over-budget description.
    # Keep the declaration as one source unit, using the same parser as coverage.
    # Full-source analysis must never extend the retrieved evidence's boundaries.
    end = item.selection.end if item.selection.end is not None else len(item.element.text)
    spans.update((declaration.start - item.selection.start, declaration.end - item.selection.start)
                 for declaration in cache.source_for(item.element.text).declarations
                 if item.selection.start <= declaration.start and declaration.end <= end
                 and declaration.end - declaration.start <= limit)
    if text.strip() and len(text) <= limit:
        spans.add((0, len(text)))
    return [build_evidence(snapshot=item.snapshot, element=item.element,
                           start=item.selection.start + start, end=item.selection.start + end)
            for start, end in sorted(spans)]


def _primary_doc_passages(
    item: EvidenceRef, task: RetrievalTask, *, limit: int,
    assessor: Callable[[RetrievalTask, list[EvidenceRef]], DocsAssessment],
    cache: DocsAssessmentCache,
) -> list[EvidenceRef]:
    """Retain an API's first complete explanation, within its retrieved range.

    A short special case can satisfy parameter coverage without preserving the
    API's defining behavior. That behavior is useful context, not a new coverage
    requirement. The existing assessor excludes declarations and release metadata.
    """
    requirement = resolve_requirement(task.requirement)
    if not requirement.symbols or not set(requirement.evidence_needs) & {"explanation", "parameters"}:
        return []
    pending = set(requirement.symbols)
    passages = []
    for section in cache.sections_for(item):
        if section.kind not in {"overview", "signature"}:
            continue
        targets = [target for target in requirement.symbols if target in pending and (
            section.owner == target or ("." not in target and section.owner.endswith("." + target))
        )]
        for target in targets:
            probe = task.model_copy(update={"requirement": requirement.model_copy(update={
                "symbols": [target], "aspects": [], "evidence_needs": ["explanation"], "match": "definition",
            })})
            # Find paragraph boundaries in the full section. Cropped leading or
            # trailing text must never become a supposedly complete description.
            for start, end in _paragraph_ranges(item.element.text[section.start:section.end]):
                start, end = section.start + start, section.start + end
                if item.element.text[start:end].lstrip().startswith(("```", "~~~", ">>>")):
                    continue
                selected = build_evidence(snapshot=item.snapshot, element=item.element, start=start, end=end)
                if assessor(probe, [selected]).answerability != "covered":
                    continue
                pending.remove(target)
                if len(selected.excerpt) <= limit and contains_evidence_range(item, selected):
                    passages.append(selected)
                break
        if not pending:
            break
    return passages


def select_evidence_packet(
    evidence: list[EvidenceRef], *, budget: RetrievedEvidenceBudget, query: str = "",
    requirements_by_evidence: dict[str, list[RetrievalTask]] | None = None,
) -> tuple[list[EvidenceRef], dict[str, list[str]]]:
    """Reserve required passages before expanding context, charging shared ranges only once."""
    packet: list[EvidenceRef] = []
    requirement_ids: dict[str, list[str]] = {}
    remaining = budget.max_total_excerpt_chars
    candidates: dict[tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], tuple[EvidenceRef, RetrievalTask | None]] = {}
    contracts: dict[str, RetrievalTask] = {}
    docs_assessments: dict[tuple[str, tuple[str, ...]], DocsAssessment] = {}
    docs_source_analysis = DocsAssessmentCache()

    def assess_selected_docs(task: RetrievalTask, selected: list[EvidenceRef]) -> DocsAssessment:
        # Fair-share and passage scoring revisit the same packet for each
        # aspect/need candidate. Evidence IDs bind immutable source ranges;
        # only reuse an identical ordered packet and complete task in this
        # selection call, never across requests or changed source selections.
        key = (task.model_dump_json(), tuple(item.id for item in selected))
        if key not in docs_assessments:
            docs_assessments[key] = assess_docs_evidence(
                task.query, selected, task.requirement, cache=docs_source_analysis)
        return docs_assessments[key]
    for item in evidence:
        tasks = (requirements_by_evidence or {}).get(item.id) or [None]
        for task in tasks:
            if task is not None:
                contracts[task.requirement_id] = task
            targets = requirement_passage_targets(task) if task is not None else [(None, None, None)]
            for file_id, aspect, need in targets:
                if file_id is not None and upload_file_id(item) != file_id:
                    continue
                changes = {}
                if (task is not None and task.route == "upload" and task.requirement.match == "topic"
                        and len(literal_passage_anchors(task)) > 1):
                    changes["symbols"] = [aspect] if aspect in task.requirement.symbols else []
                    changes["aspects"] = [aspect] if aspect in task.requirement.aspects else []
                elif task is not None and len(task.requirement.aspects) > 1:
                    changes["aspects"] = [aspect]
                if task is not None and len(task.requirement.evidence_needs) > 1:
                    changes["evidence_needs"] = [need]
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
            associated = [selected for selected in packet if task.requirement_id in requirement_ids.get(selected.id, [])]
            assessment = assess_selected_docs(contracts[task.requirement_id], associated)
            return bool(_focused_docs_missing(assessment, task))
        selected_ranges = [selected for selected in packet
                    if (task is None and selected.route == item.route) or (
                        task is not None and task.requirement_id in requirement_ids.get(selected.id, [])
                        and (not task.requirement.file_ids or upload_file_id(selected) == upload_file_id(item)))]
        return not selected_ranges or bool(task and missing_literal_aspects(literal_passage_anchors(task), selected_ranges))

    # All associations are attached before testing capacity: sharing a cropped
    # range is free even when the item or character budget has been exhausted.
    def share_existing(item: EvidenceRef, task: RetrievalTask | None) -> None:
        for selected in packet:
            if contains_evidence_range(item, selected):
                associate(selected, task)

    def replaced_ranges(selected: EvidenceRef) -> list[EvidenceRef]:
        return [old for old in packet if contains_evidence_range(selected, old)
                and all(any(candidate_task is not None and candidate_task.requirement_id == requirement_id
                            and contains_evidence_range(candidate, selected)
                            for candidate, candidate_task in ordered)
                        for requirement_id in requirement_ids.get(old.id, []))]

    def admit(selected: EvidenceRef, item: EvidenceRef, task: RetrievalTask | None) -> bool:
        nonlocal remaining
        for old in packet:
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

    # Docs aspects and evidence kinds are independent coverage obligations,
    # not an aspect x kind budget. Reserve complete useful passages before any
    # optional context; inspect the full immutable task for every proposal.
    doc_proposals: dict[tuple[str, str], tuple[EvidenceRef, EvidenceRef, RetrievalTask]] = {}
    seen_docs = set()
    for item, focused in ordered:
        if focused is None or focused.route != "docs" or (item.id, focused.requirement_id) in seen_docs:
            continue
        seen_docs.add((item.id, focused.requirement_id))
        task = contracts[focused.requirement_id]
        if item.element.kind == "table":
            selected = select_table_evidence(item, limit=budget.max_excerpt_chars, query=query, task=task)
            proposals = [selected] if selected is not None else []
        else:
            proposals = _complete_doc_passages(item, budget.max_excerpt_chars, cache=docs_source_analysis)
        for selected in proposals:
            doc_proposals.setdefault((selected.id, task.requirement_id), (selected, item, task))

    def reserve_docs(allowances: dict[str, int] | None = None) -> None:
        spent: dict[str, int] = {}
        while True:
            # A source selected for another requirement may already supply this
            # one. Attach that free coverage before scoring cost or capacity.
            for item, task in ordered:
                share_existing(item, task)
            missing = {key: set(assess_selected_docs(task, [item for item in packet
                if key in requirement_ids.get(item.id, [])]).missing)
                for key, task in contracts.items() if task.route == "docs"}
            best = None
            best_score = None
            for selected, original, task in doc_proposals.values():
                key = task.requirement_id
                if not missing[key]:
                    continue
                replaced = replaced_ranges(selected)
                cost = len(selected.excerpt) - sum(len(old.excerpt) for old in replaced)
                if (cost > remaining or len(packet) - len(replaced) >= budget.max_items
                        or (allowances is not None and cost > allowances[key] - spent.get(key, 0))):
                    continue
                associated = [item for item in packet if key in requirement_ids.get(item.id, [])]
                after = set(assess_selected_docs(task, [*associated, selected]).missing)
                gain = missing[key] - after
                if not gain or after - missing[key]:
                    continue
                # Both characters and item slots are finite. Optimizing only
                # characters lets cheap generic descriptions fill every slot
                # before a passage supplying the requested parameter is chosen.
                capacity = remaining if allowances is None else min(
                    remaining, allowances[key] - spent.get(key, 0))
                pressure = max(
                    max(0, cost) / max(1, capacity),
                    max(0, 1 - len(replaced)) / max(1, budget.max_items - len(packet)),
                    1 / max(1, budget.max_total_excerpt_chars),
                )
                score = (len(gain) / pressure, len(gain), -cost)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (selected, original, task, cost)
            if best is None:
                return
            selected, original, task, cost = best
            if not admit(selected, original, task):
                return
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
                coverage_task=contracts.get(task.requirement_id) if task is not None else None,
                covered_evidence=[selected for selected in packet
                                  if task is not None and task.route == "docs"
                                  and task.requirement_id in requirement_ids.get(selected.id, [])],
                docs_assessor=assess_selected_docs,
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
        if any(original.id == item.id for original, _ in origins.values()):
            continue
        add_candidate(item, task, remaining, require_anchor=False)

    # Coverage can be complete from a short special case while the retrieved
    # API introduction has been dropped. Preserve that first explanation after
    # required and additional relevant passages, before optional neighbors.
    seen_primary = set()
    for item, focused in ordered:
        if focused is None or focused.route != "docs" or item.element.kind == "table":
            continue
        key = (item.id, focused.requirement_id)
        if key in seen_primary:
            continue
        seen_primary.add(key)
        task = contracts[focused.requirement_id]
        for selected in _primary_doc_passages(
            item, task, limit=budget.max_excerpt_chars, assessor=assess_selected_docs, cache=docs_source_analysis,
        ):
            admit(selected, item, task)

    reserved = list(packet)
    for index, selected in enumerate(reserved):
        # Expanding an earlier passage may already have subsumed this range.
        if selected.id not in origins:
            continue
        original, task = origins[selected.id]
        if original.element.kind == "table":
            continue
        limit = min(budget.max_excerpt_chars, len(selected.excerpt) + remaining // (len(reserved) - index))
        expanded = select_evidence_range(
            original, limit=limit, query=query, task=task,
            coverage_task=contracts.get(task.requirement_id) if task is not None else None,
            docs_assessor=assess_selected_docs,
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
    requirement_ids.clear()
    for item, task in ordered:
        share_existing(item, task)
    return packet, requirement_ids
