from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

from src.core.evidence import EvidenceRef, SearchHit, build_evidence
from src.core.planner_schema import RetrievalRequirement
from src.infra.tools.docs_search.policy import result_matches_domains
from src.infra.tools.docs_search.requirements import library_domains, resolve_requirement
from src.infra.tools.docs_search.sections import DocDeclaration, DocSection, clean_markup, parse_doc_sections, scan_declarations


_TOPIC_STOPWORDS = {
    "official", "docs", "documentation", "reference", "api", "parameters", "parameter",
    "params", "arguments", "argument", "args", "options", "option", "signature",
    "examples", "example", "explain", "explanation", "describe", "show", "tell",
    "how", "what", "the", "a", "an", "and", "or", "of", "for", "with", "to", "in",
    "is", "are", "using", "use", "usage", "definition", "function", "method", "class",
    "latest", "stable", "current", "please", "code", "공식", "문서", "옵션", "설명",
    "매개변수", "파라미터", "인자", "예제", "예시", "사용", "사용법", "알려줘", "정리해줘",
    "보여줘", "정리", "코드", "차트", "기준", "특정", "찾는",
    "으로", "에서", "대한", "대해서", "어떻게", "무엇", "함수", "메서드",
}


def _topic_terms(query: str, library: str) -> set[str]:
    ignored = _TOPIC_STOPWORDS | set(re.findall(r"[A-Za-z0-9_-]+", library.casefold()))
    ignored |= {"np", "pd", "plt", "sklearn", "torch", "bs4", "st"}
    terms = {token.casefold() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]*", query)
             if token.casefold() not in ignored and not re.fullmatch(r"v?\d+(?:\.\d+)*", token)}
    terms.update(token for token in re.findall(r"[가-힣]{2,}", query)
                 if not any(token.startswith(stop) for stop in ignored if re.fullmatch(r"[가-힣]+", stop)))
    return terms


Coverage = tuple[str, str, str]
_WINDOW_CHARS = 3200


@dataclass(frozen=True)
class DocsAssessment:
    hits: list[SearchHit]
    answerability: Literal["covered", "partial", "missing"]
    missing: list[str]
    rejected_symbols: int = 0


@dataclass(frozen=True)
class _Support:
    coverage: frozenset[Coverage]
    relevance: float
    confidence: int
    version: str
    unresolved_topic: bool = False
    primary_explanations: int = 0


@dataclass(frozen=True)
class _Candidate:
    hit: SearchHit
    support: _Support


@dataclass(frozen=True)
class _SourceAnalysis:
    declarations: list[DocDeclaration]
    prose_source: str


@dataclass
class DocsAssessmentCache:
    """Memoized immutable source analysis for one evidence-selection operation."""

    sections: dict[tuple[str, str, str], list[DocSection]] = field(default_factory=dict)
    supports: dict[tuple[str, str, tuple[str, ...], tuple[str, ...], str], _Support | None] = field(default_factory=dict)
    sources: dict[str, _SourceAnalysis] = field(default_factory=dict)

    def source_for(self, text: str) -> _SourceAnalysis:
        if text not in self.sources:
            declarations = scan_declarations(text)
            # Recognition-only view: keep source offsets and line boundaries,
            # but exclude declarations/types/defaults from descriptive prose.
            parts: list[str] = []
            previous_end = 0
            for declaration in declarations:
                parts.append(text[previous_end:declaration.start])
                parts.append(re.sub(r"[^\r\n]", " ", text[declaration.start:declaration.end]))
                previous_end = declaration.end
            parts.append(text[previous_end:])
            self.sources[text] = _SourceAnalysis(declarations, "".join(parts))
        return self.sources[text]

    def sections_for(self, evidence: EvidenceRef) -> list[DocSection]:
        key = (evidence.element.text, evidence.snapshot.title, evidence.snapshot.source_uri)
        if key not in self.sections:
            self.sections[key] = parse_doc_sections(
                key[0], title=key[1], url=key[2], declarations=self.source_for(key[0]).declarations,
            )
        return self.sections[key]


def _contains(text: str, term: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", clean_markup(text), re.I) is not None


def _owns(owner: str, target: str) -> bool:
    return bool(owner) and (owner == target or ("." not in target and owner.endswith("." + target)))


def _topic_contains(text: str, term: str) -> bool:
    if _contains(text, term):
        return True
    # Ordinary English topic words may be plural; API/aspect identity remains exact.
    if not re.fullmatch(r"[a-z]{4,}", term):
        return False
    singular = term[:-1] if term.endswith("s") else term
    return _contains(text, singular) or _contains(text, singular + "s")


def _version_state(evidence: EvidenceRef, wanted: str | None) -> str:
    if not wanted:
        return "match"
    path = urlparse(evidence.snapshot.source_uri).path
    title = evidence.snapshot.title
    if wanted.casefold() in {"stable", "latest", "current"}:
        return "match" if re.search(r"/(?:latest|stable)/", path, re.I) else "unknown"
    version = wanted.removeprefix("v")
    versions = re.findall(r"/(?:v)?(\d+(?:\.\d+)*)(?=/|$)", path)
    versions += re.findall(r"\bv?(\d+\.\d+(?:\.\d+)*)\b", title)
    if any(value == version or value.startswith(version + ".") for value in versions):
        return "match"
    return "mismatch" if versions else "unknown"


def _meaningful_lines(text: str) -> list[str]:
    headings = {"parameters", "parameter", "options", "returns", "examples", "example", "notes",
                "arguments", "args", "signature", "overview", "description", "see also"}
    return [line for raw in text.splitlines()
            if (line := clean_markup(raw).lstrip("#: ").rstrip()) and line.casefold().rstrip(":") not in headings
            and not raw.lstrip().startswith(("#", "```", "~~~"))]


_TYPE_START = re.compile(
    r"^(?:\d+D\b|array[ _-]?like\b|ndarray\b|bool(?:ean)?\b|int(?:eger)?\b|float\b|str(?:ing)?\b|"
    r"bytes\b|list\b|dict\b|tuple\b|set\b|sequence\b|iterable\b|callable\b|object\b|dtype\b|"
    r"None\b|optional\b|\{|\[)", re.I,
)


def _parameter_entries(text: str) -> dict[str, str]:
    """Bind a named entry to its description, not mentions in another entry."""
    described: dict[str, str] = {}
    current = ""
    for raw in text.splitlines():
        line = clean_markup(raw).lstrip("#: ").strip()
        if not line or line.casefold().rstrip(":") in {"parameters", "options", "args", "arguments"}:
            continue
        bold = re.match(r"^\s*(?::\s*)?\*\*(?P<name>(?:\\.|[^*])+)\*\*(?P<tail>.*)$", raw)
        if bold and re.fullmatch(r"(?:--?|\*{0,2})?[A-Za-z_][\w-]*", clean_markup(bold.group("name"))):
            current = clean_markup(bold.group("name")).lstrip("*")
            # Sphinx places arbitrary type names immediately after the bold
            # parameter. That type is not its description, even if unfamiliar.
            described[current] = ""
            tail = clean_markup(bold.group("tail"))
            if tail.startswith(":"):
                tail = tail.lstrip(": ")
                if not _TYPE_START.match(tail) and len(re.findall(r"[A-Za-z가-힣]+", tail)) >= 2:
                    described[current] = tail
            continue
        if line.startswith("|"):
            cells = [cell.strip(" `") for cell in line.strip("|").split("|")]
            if len(cells) >= 3 and re.fullmatch(r"[A-Za-z_]\w*", cells[0]) and len(cells[-1].split()) >= 3:
                described[cells[0]] = cells[-1]
            current = ""
            continue
        entry = re.match(r"^(?P<name>(?:--?|\*{0,2})[A-Za-z_][\w-]*|[A-Za-z_]\w*)\s*:\s*(?P<tail>.+)$", line)
        if entry is None:
            typed = re.match(r"^(?P<name>[A-Za-z_]\w*)\s+(?P<tail>.+)$", line)
            if typed and _TYPE_START.match(typed.group("tail")):
                entry = typed
        if entry is not None:
            current = entry.group("name").lstrip("*")
            tail = entry.group("tail")
            described[current] = ""
            if not _TYPE_START.match(tail) and len(re.findall(r"[A-Za-z가-힣]+", tail)) >= 2:
                described[current] = tail
        elif current and len(re.findall(r"[A-Za-z가-힣]+", line)) >= 3:
            described[current] = " ".join([described.get(current, ""), line]).strip()
    return described


def _declared_parameter_names(
    text: str, sections: list[DocSection], owner: str, declarations: list[DocDeclaration],
) -> set[str]:
    """Distinguish parameter names from value anchors, without granting coverage."""
    names: set[str] = set()
    for section in sections:
        if section.owner != owner:
            continue
        source = text[section.start:section.end]
        if section.kind in {"parameters", "options"}:
            names.update(_parameter_entries(source))
        elif section.kind == "signature":
            for declaration in declarations:
                if (section.start <= declaration.start < declaration.end <= section.end
                        and (not owner or owner == declaration.name or owner.endswith("." + declaration.name))):
                    names.update(declaration.parameters)
    return names


def _example_content(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*(?:>>>|\w[\w.]*\s*\(|\w+\s*=|import\s|from\s)", text))


def _usage(text: str, target: str) -> bool:
    # Resolve only namespace aliases; never introduce per-function equivalences.
    aliases = {"np": "numpy", "pd": "pandas", "plt": "matplotlib.pyplot", "st": "streamlit"}
    for alias, namespace in aliases.items():
        text = re.sub(rf"(?<![\w.]){alias}\.", namespace + ".", text)
    return re.search(rf"(?<![A-Za-z0-9_.]){re.escape(target)}\s*\(", text) is not None


_RELEASE_PREFIX = re.compile(
    r"^(?:changed|new|added|deprecated|removed)\s+(?:in|since)\s+version\s+[\w.]+\s*:?\s*",
    re.I,
)
_LIFECYCLE_STATEMENT = re.compile(
    r"^[\w .,/-]+\s+(?:was|were|is|are|has been|have been)\s+"
    r"(?:added|introduced|deprecated|removed)(?:\s+in\s+(?:version\s+)?[\w.]+)?[.!]?$",
    re.I,
)
_API_LIFECYCLE_SUBJECT = re.compile(
    r"^(?:[\w.]+\s+)*(?:parameter|argument|keyword|option|flag|function|method|class|attribute|module|alias|API)s?\s+",
    re.I,
)
_KOREAN_PARAMETER_SUBJECT = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:은|는|이|가)\s+(?P<body>.+)$",
)
_KOREAN_LIFECYCLE_BODY = re.compile(
    r"(?:(?:버전\s*)?\d+(?:\.\d+)*\s*(?:버전)?(?:에서|에|부터)\s+)?"
    r"(?:(?:새로운|새|선택적)\s+)*(?:(?:매개변수|인자|인수|옵션)(?:로|으로)\s+)?"
    r"(?:추가|도입|제거|삭제|폐기|지원\s*중단)(?:되었습니다|됐습니다|됩니다|됨|되었음)[.!]?",
)


def _korean_lifecycle_statement(text: str) -> bool:
    subject = _KOREAN_PARAMETER_SUBJECT.match(text)
    return bool(subject and _KOREAN_LIFECYCLE_BODY.fullmatch(subject.group("body")))


def _explanation_lines(text: str) -> list[str]:
    """Keep descriptive prose, without mistaking API lifecycle metadata for it.

    Release notices remain in the source and can contain real behavior. Strip
    their annotation before checking the body: a return/exception condition is
    useful, while saying only that an argument was added does not explain it.
    Applying the same check to continuation lines handles wrapped notices.
    """
    result = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = _meaningful_lines(paragraph)
        statement = _RELEASE_PREFIX.sub("", " ".join(lines))
        release_note = bool(lines and _RELEASE_PREFIX.match(lines[0]))
        if not statement or _korean_lifecycle_statement(statement) or ((release_note or _API_LIFECYCLE_SUBJECT.match(statement))
                             and _LIFECYCLE_STATEMENT.fullmatch(statement)):
            continue
        for original in lines:
            line = _RELEASE_PREFIX.sub("", original)
            if _korean_lifecycle_statement(line) or ((release_note or _RELEASE_PREFIX.match(original) or _API_LIFECYCLE_SUBJECT.match(line))
                    and _LIFECYCLE_STATEMENT.fullmatch(line)):
                continue
            if (len(re.findall(r"[A-Za-z가-힣]+", line)) >= 4
                    and not line.endswith(":")
                    and not line.startswith((">>>", "import ", "from "))
                    and not re.match(r"^@?[A-Za-z_][\w.]*\s*\(", line)):
                result.append(line)
    return result


def _prose_parameters(text: str, declared: set[str]) -> dict[str, str]:
    """Bind behavior to a declared parameter that is the sentence's subject.

    A mention inside another parameter's description is not an entry. Release
    metadata and signatures likewise cannot supply parameter behavior.
    """
    descriptions: dict[str, str] = {}
    for line in _explanation_lines(text):
        subject = re.match(
            r"^(?:(?:if|when|unless)\s+)?(?:the\s+)?(?P<name>[A-Za-z_]\w*)\s+"
            r"(?:(?:parameter|argument|option)\s+)?"
            r"(?:is|are|has|have|must|can|may|should|will|controls?|determines?|specifies|"
            r"sets?|limits?|defaults?|enables?|disables?|selects?|indicates?)\b",
            line, re.I,
        )
        if subject is None:
            # Translated references attach a Korean subject/topic particle to
            # the unchanged parameter name instead of using an English verb.
            subject = _KOREAN_PARAMETER_SUBJECT.match(line)
        if subject and subject.group("name") in declared:
            name = subject.group("name")
            descriptions[name] = " ".join([descriptions.get(name, ""), line]).strip()
    return descriptions


def _section_needs(
    text: str, section: DocSection, parameters: dict[str, str], *, has_signature: bool, prose: str,
) -> set[str]:
    needs: set[str] = set()
    if has_signature:
        needs.add("signature")
    if parameters:
        needs.add("parameters")
    if section.kind == "examples" and _example_content(text):
        needs.add("examples")
    if _explanation_lines(prose):
        needs.add("explanation")
    return needs


def _required_coverage(requirement: RetrievalRequirement) -> set[Coverage]:
    result: set[Coverage] = set()
    for target in requirement.symbols or [""]:
        result.add(("symbol" if target else "topic", target, ""))
        result.update(("aspect", target, aspect) for aspect in requirement.aspects)
        result.update(("need", target, need) for need in requirement.evidence_needs)
    if requirement.version:
        result.add(("version", "", requirement.version))
    return result


def _candidate_windows(hit: SearchHit, sections: list[DocSection], requirement: RetrievalRequirement) -> list[SearchHit]:
    """Offer original and focused ranges before the final k budget is consumed."""
    text = hit.evidence.element.text
    ranges = {(hit.evidence.selection.start, hit.evidence.selection.end or len(text))}
    if len(text) <= _WINDOW_CHARS:
        ranges.add((0, len(text)))
    else:
        for section in sections:
            if requirement.symbols and section.owner and not any(_owns(section.owner, s) for s in requirement.symbols):
                # Usage examples can have another owner and are checked separately.
                if section.kind != "examples":
                    continue
            targets = [section.start]
            for aspect in requirement.aspects:
                targets.extend(section.start + match.start() for match in re.finditer(
                    r"(?<![A-Za-z0-9_])" + re.escape(aspect).replace("_", r"\\?_") + r"(?![A-Za-z0-9_])",
                    text[section.start:section.end], re.I,
                ))
            for position in targets:
                start = max(section.start, min(position - 400, section.end - _WINDOW_CHARS))
                # Round forward when possible: rounding a tail window backward
                # would cut off the last description to retain unrelated context.
                next_line = text.find("\n", start, position) + 1
                start = next_line if next_line > start else max(section.start, text.rfind("\n", section.start, start) + 1)
                ranges.add((start, min(section.end, start + _WINDOW_CHARS)))
    return [hit.model_copy(update={"evidence": build_evidence(
        snapshot=hit.evidence.snapshot, element=hit.evidence.element, start=start, end=end,
    )}) for start, end in sorted(ranges) if end > start and end - start <= _WINDOW_CHARS]


def _complete_source_lines(source: str, start: int, end: int) -> str:
    """A description cut inside a source line cannot establish its full condition."""
    if start and source[start - 1] != "\n":
        boundary = source.find("\n", start, end)
        if boundary < 0:
            return ""
        start = boundary + 1
    if end < len(source) and source[end:end + 1] != "\n" and source[end - 1:end] != "\n":
        boundary = source.rfind("\n", start, end)
        if boundary < 0:
            return ""
        end = boundary + 1
    return source[start:end]


def _selected_declaration(source: str, declaration: DocDeclaration, start: int, end: int) -> bool:
    """Require the source declaration, allowing only omitted markup at its edges."""
    if declaration.end <= start or end <= declaration.start:
        return False
    omitted_prefix = source[declaration.start:max(declaration.start, start)]
    omitted_suffix = source[min(declaration.end, end):declaration.end]
    # Definition/self-links, heading markers and code markup are metadata.
    # A cropped nested call still omits the outer declaration's actual text.
    return not any(clean_markup(edge).strip("# ¶ ") for edge in (omitted_prefix, omitted_suffix))


def _evaluate_window(
    evidence: EvidenceRef, sections: list[DocSection], requirement: RetrievalRequirement,
    subjects: set[str], focus: set[str], version: str, source: _SourceAnalysis,
) -> _Support | None:
    if not _has_grounded_text(evidence):
        return None
    start, end = evidence.selection.start, evidence.selection.end
    end = len(evidence.element.text) if end is None else end
    declarations, prose_source = source.declarations, source.prose_source
    parts = [(section, evidence.element.text[max(start, section.start):min(end, section.end)])
             for section in sections if section.start < end and start < section.end]
    coverage: set[Coverage] = set()
    primary_explanations: set[str] = set()
    confidence = 0
    matched_terms: set[str] = set()
    usage_allowed = requirement.match != "definition" and any(
        need in requirement.evidence_needs for need in ("examples", "explanation")
    )
    unresolved_topic = False
    declared_by_owner: dict[str, set[str]] = {}
    for target in requirement.symbols or [""]:
        for section, text in parts:
            if not _meaningful_lines(text):
                continue
            owned = _owns(section.owner, target) if target else False
            used = section.kind == "examples" and usage_allowed and _usage(text, target) if target else False
            if target and not (owned or used):
                continue
            if not target:
                haystack = f"{section.owner} {evidence.snapshot.title} {text}"
                matched = {term for term in subjects if _topic_contains(haystack, term)}
                if subjects and not matched:
                    # Preserve potentially useful cross-language evidence, but do
                    # not confuse an unverified translation with topic coverage.
                    if all(re.fullmatch(r"[가-힣]+", term) for term in subjects):
                        unresolved_topic = True
                    continue
                # A discovered short API name is the subject, not incidental prose
                # in another API's parameter table on the same page.
                if focus and section.owner and not any(_owns(section.owner, term) for term in focus):
                    if not (usage_allowed and section.kind == "examples"
                            and any(_usage(text, term) for term in focus)):
                        continue
                matched_terms.update(matched)
            coverage.add(("symbol" if target else "topic", target, ""))
            confidence = max(confidence, section.confidence)
            part_start, part_end = max(start, section.start), min(end, section.end)
            prose = prose_source[part_start:part_end]
            parameter_text = _complete_source_lines(prose_source, part_start, part_end)
            if section.owner not in declared_by_owner:
                declared_by_owner[section.owner] = _declared_parameter_names(
                    evidence.element.text, sections, section.owner, declarations,
                )
            parameter_names = declared_by_owner[section.owner]
            parameters: dict[str, str] = {}
            if section.kind in {"parameters", "options", "overview", "signature"}:
                entries = _parameter_entries(parameter_text)
                if section.kind == "signature":
                    # A label in the definition's prose (such as "Note:") is
                    # not a parameter table. Its names require a declaration.
                    entries = {name: description for name, description in entries.items() if name in parameter_names}
                parameters = {name: description for name, description in entries.items() if description}
                if section.kind in {"overview", "signature"} and not entries:
                    parameters.update(_prose_parameters(parameter_text, parameter_names))
            has_signature = section.kind == "signature" and any(
                _selected_declaration(evidence.element.text, declaration, part_start, part_end)
                and (not section.owner or section.owner == declaration.name or section.owner.endswith("." + declaration.name))
                for declaration in declarations
            )
            needs = _section_needs(text, section, parameters, has_signature=has_signature, prose=prose)
            if ("explanation" in requirement.evidence_needs and "explanation" in needs
                    and ("signature" in needs or section.kind == "overview")):
                primary_explanations.add(target)
            if target and "examples" in needs and not (
                _usage(text, target) or (owned and (
                    _usage(text, section.owner) or _usage(text, target.rsplit(".", 1)[-1])
                ))
            ):
                needs.discard("examples")
            if not target and focus and "examples" in needs and not any(
                _usage(text, term) or (_owns(section.owner, term) and _usage(text, section.owner))
                for term in focus
            ):
                needs.discard("examples")
            if used and not owned:
                needs &= {"examples", "explanation"}
            coverage.update(("need", target, need) for need in requirement.evidence_needs if need in needs)
            requested_descriptions = [description for name, description in parameters.items()
                                      if name in requirement.aspects]
            for aspect in requirement.aspects:
                if "parameters" in requirement.evidence_needs:
                    if "parameters" not in needs:
                        continue
                    if aspect not in parameters and (aspect in parameter_names or not any(
                        _contains(description, aspect) for description in requested_descriptions
                    )):
                        continue
                if "explanation" in requirement.evidence_needs and aspect not in parameters and not any(
                    _contains(line, aspect) for line in _explanation_lines(prose)
                ):
                    # A name in the signature or release history does not
                    # explain that parameter's behavior.
                    continue
                if _contains(text, aspect):
                    coverage.add(("aspect", target, aspect))
    if not coverage and not unresolved_topic:
        return None
    if requirement.version and version == "match":
        coverage.add(("version", "", requirement.version))
    relevance = len(matched_terms) / max(1, len(subjects)) if not requirement.symbols else 1.0
    return _Support(frozenset(coverage), relevance, confidence, version, unresolved_topic, len(primary_explanations))


def _missing_labels(required: set[Coverage], covered: set[Coverage], requirement: RetrievalRequirement) -> list[str]:
    order = {"symbol": 0, "version": 1, "aspect": 2, "need": 3, "topic": 4}
    labels = []
    for kind, target, value in sorted(required - covered, key=lambda item: (order[item[0]], item[1], item[2])):
        if kind == "symbol":
            labels.append(f"symbol:{target}")
        elif kind == "topic":
            labels.append("topic")
        else:
            suffix = f":{target}" if target and len(requirement.symbols) > 1 else ""
            labels.append(f"{kind}:{value}{suffix}")
    return labels


def _assessment(
    supports: list[_Support], requirement: RetrievalRequirement, *,
    hits: list[SearchHit], rejected_symbols: int = 0,
) -> DocsAssessment:
    required = _required_coverage(requirement)
    verified = set().union(*(support.coverage for support in supports if support.version == "match"))
    missing = _missing_labels(required, verified, requirement)
    if requirement.version and any(support.version == "unknown" for support in supports) and missing:
        label = f"version:{requirement.version}"
        if label not in missing:
            missing.insert(0, label)
    return DocsAssessment(
        hits=hits, answerability="partial" if missing and supports else "missing" if missing else "covered",
        missing=missing, rejected_symbols=rejected_symbols,
    )


def assess_docs_evidence(
    query: str, evidences: list[EvidenceRef], requirement: RetrievalRequirement,
    *, cache: DocsAssessmentCache | None = None,
) -> DocsAssessment:
    """Assess only the supplied source ranges, without expanding or selecting them.

    Full source structure supplies ownership and section boundaries; only text
    inside each selection supplies factual coverage. This also checks the final
    cropped ranges sent to answer generation with the retrieval contract.
    """
    requirement = resolve_requirement(requirement)
    domains = library_domains(str(requirement.library or ""))
    if requirement.library and not domains:
        return DocsAssessment(hits=[], answerability="missing", missing=[f"library:{requirement.library}"])
    subjects = _topic_terms(query, str(requirement.library or ""))
    cache = cache if cache is not None else DocsAssessmentCache()
    parsed = [(evidence, cache.sections_for(evidence))
              for evidence in evidences if evidence.route == "docs"
              and result_matches_domains(evidence.snapshot.source_uri, set(domains))]
    focus = {term for term in subjects for _, sections in parsed for section in sections
             if section.owner and _owns(section.owner, term)}
    supports: list[_Support] = []
    rejected = 0
    for evidence, sections in parsed:
        version = _version_state(evidence, requirement.version)
        if version == "mismatch":
            continue
        key = (evidence.id, requirement.model_dump_json(), tuple(sorted(subjects)), tuple(sorted(focus)), version)
        if key in cache.supports:
            support = cache.supports[key]
        else:
            support = _evaluate_window(
                evidence, sections, requirement, subjects, focus, version,
                cache.source_for(evidence.element.text),
            )
            cache.supports[key] = support
        if support is not None:
            supports.append(support)
        elif requirement.symbols:
            rejected += 1
    return _assessment(supports, requirement, hits=[], rejected_symbols=rejected)


def select_docs_evidence(
    query: str, hits: list[SearchHit], requirement: RetrievalRequirement, *,
    k: int, allowed_domains: list[str], retrieval_warnings: list[str],
    preferred_snapshot_ids: set[str] | None = None,
) -> DocsAssessment:
    """One scope/coverage model drives ranking, evidence selection and completion.

    Document formatting and provider scores cannot compensate for a different
    target. Only exact returned source ranges contribute to required coverage.
    """
    required = _required_coverage(requirement)
    subjects = _topic_terms(query, str(requirement.library or ""))
    cache = DocsAssessmentCache()
    parsed = [(hit, cache.sections_for(hit.evidence))
              for hit in dedupe_docs_hits(hits)
              if result_matches_domains(hit.evidence.snapshot.source_uri, set(allowed_domains))]
    focus = {term for term in subjects for _, sections in parsed for section in sections
             if section.owner and _owns(section.owner, term)}
    candidates: list[_Candidate] = []
    rejected = 0
    for hit, sections in parsed:
        version = _version_state(hit.evidence, requirement.version)
        if version == "mismatch":
            continue
        source = cache.source_for(hit.evidence.element.text)
        evaluated = [_Candidate(window, support) for window in _candidate_windows(hit, sections, requirement)
                     if (support := _evaluate_window(window.evidence, sections, requirement, subjects, focus, version, source))]
        if not evaluated:
            if requirement.symbols:
                rejected += 1
            if not _has_grounded_text(hit.evidence):
                retrieval_warnings.append("docs_chrome_only")
        candidates.extend(evaluated)

    selected: list[SearchHit] = []
    selected_supports: list[_Support] = []
    covered: set[Coverage] = set()
    verified: set[Coverage] = set()
    budget = min(16000, max(1, k) * _WINDOW_CHARS)
    # Stable source order breaks otherwise equal ties independently of provider order.
    pending = sorted(candidates, key=lambda c: (c.hit.evidence.snapshot.source_uri, c.hit.evidence.selection.start))
    while pending and len(selected) < k:
        fitting = [c for c in pending if len(c.hit.evidence.excerpt) <= budget
                   and (c.support.coverage - covered or (not selected and c.support.unresolved_topic))]
        if not fitting:
            break
        best = max(fitting, key=lambda c: (
            c.support.version == "match",
            len((c.support.coverage & required) - covered),
            c.support.relevance,
            c.support.confidence,
            # For a behavioral question, the definition's own description
            # precedes shorter incidental notes with otherwise equal coverage.
            c.support.primary_explanations,
            # On an explicit refresh, equally scoped new source revisions get
            # a chance before the provider score of previously rejected text.
            c.hit.evidence.snapshot.snapshot_id in (preferred_snapshot_ids or ()),
            float(c.hit.score.normalized or 0.0),
            -len(c.hit.evidence.excerpt),
        ))
        selected.append(best.hit)
        selected_supports.append(best.support)
        covered.update(best.support.coverage)
        if best.support.version == "match":
            verified.update(best.support.coverage)
        budget -= len(best.hit.evidence.excerpt)
        pending = [c for c in pending if c.hit.evidence.id != best.hit.evidence.id]
        if required <= verified:
            break
    return _assessment(selected_supports, requirement, hits=selected, rejected_symbols=rejected)


def _has_grounded_text(evidence: EvidenceRef) -> bool:
    """Check returned source text, never treat a document title as factual support."""
    navigation_prefixes = (
        "table of contents",
        "on this page",
        "previous:",
        "next:",
        "skip to content",
        "edit this page",
        "view source",
        "home >",
        "navigation",
        "back to top",
    )
    section_names = {"parameters", "returns", "examples", "notes", "contents", "api reference", "reference"}
    for raw_line in evidence.excerpt.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.lower().startswith(navigation_prefixes):
            continue
        if line.rstrip(":").lower() in section_names:
            continue
        if re.fullmatch(r"\[[^]]+\]\([^)]+\)", line):
            continue
        if line.lower() == evidence.snapshot.title.lower():
            continue
        if re.search(r"[A-Za-z가-힣0-9]", line):
            return True
    return False


def dedupe_docs_hits(items: list[SearchHit]) -> list[SearchHit]:
    """Deduplicate exact source selections without concatenating different excerpts."""
    by_evidence: dict[str, SearchHit] = {}
    for hit in items:
        key = hit.evidence.id
        current = by_evidence.get(key)
        score = hit.score.normalized if hit.score else None
        current_score = current.score.normalized if current and current.score else None
        if current is None or (score is not None and (current_score is None or score > current_score)):
            by_evidence[key] = hit
    return [hit.model_copy(update={"rank": index}) for index, hit in enumerate(by_evidence.values(), start=1)]
