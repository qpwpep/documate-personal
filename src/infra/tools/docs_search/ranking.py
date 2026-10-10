from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from src.core.evidence import EvidenceRef, SearchHit, build_evidence
from src.core.planner_schema import RetrievalRequirement
from src.infra.tools.docs_search.policy import result_matches_domains
from src.infra.tools.docs_search.requirements import library_domains, resolve_requirement
from src.infra.tools.docs_search.sections import DocSection, clean_markup, parse_doc_sections


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


@dataclass(frozen=True)
class _Candidate:
    hit: SearchHit
    support: _Support


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
            if (line := clean_markup(raw).strip("#: ")) and line.casefold() not in headings
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


def _described_parameters(text: str) -> dict[str, str]:
    return {name: description for name, description in _parameter_entries(text).items() if description}


def _declared_parameter_names(text: str, sections: list[DocSection], owner: str) -> set[str]:
    """Distinguish parameter names from value anchors, without granting coverage."""
    names: set[str] = set()
    for section in sections:
        if section.owner != owner:
            continue
        source = text[section.start:section.end]
        if section.kind in {"parameters", "options"}:
            names.update(_parameter_entries(source))
        elif section.kind == "signature":
            signature = clean_markup(source)
            start, end = signature.find("("), signature.rfind(")")
            if start >= 0 and end > start:
                names.update(re.findall(r"(?:^|,)\s*\*{0,2}([A-Za-z_]\w*)", signature[start + 1:end]))
    return names


def _example_content(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*(?:>>>|\w[\w.]*\s*\(|\w+\s*=|import\s|from\s)", text))


def _usage(text: str, target: str) -> bool:
    # Resolve only namespace aliases; never introduce per-function equivalences.
    aliases = {"np": "numpy", "pd": "pandas", "plt": "matplotlib.pyplot", "st": "streamlit"}
    for alias, namespace in aliases.items():
        text = re.sub(rf"(?<![\w.]){alias}\.", namespace + ".", text)
    return re.search(rf"(?<![A-Za-z0-9_.]){re.escape(target)}\s*\(", text) is not None


def _section_needs(text: str, section: DocSection, parameter_text: str) -> set[str]:
    needs: set[str] = set()
    lines = _meaningful_lines(text)
    if section.kind == "signature" and re.search(r"[A-Za-z_]\w*(?:\.\w+)*\s*\([^\n]*\)", clean_markup(text)):
        needs.add("signature")
    if section.kind in {"parameters", "options", "overview"} and _described_parameters(parameter_text):
        needs.add("parameters")
    if section.kind == "examples" and _example_content(text):
        needs.add("examples")
    if any(len(re.findall(r"[A-Za-z가-힣]+", line)) >= 4
           and not line.startswith((">>>", "import ", "from "))
           and not re.match(r"^[A-Za-z_][\w.]*\s*\(", line) for line in lines):
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


def _evaluate_window(
    evidence: EvidenceRef, sections: list[DocSection], requirement: RetrievalRequirement,
    subjects: set[str], focus: set[str], version: str,
) -> _Support | None:
    if not _has_grounded_text(evidence):
        return None
    start, end = evidence.selection.start, evidence.selection.end
    end = len(evidence.element.text) if end is None else end
    parts = [(section, evidence.element.text[max(start, section.start):min(end, section.end)])
             for section in sections if section.start < end and start < section.end]
    coverage: set[Coverage] = set()
    confidence = 0
    matched_terms: set[str] = set()
    usage_allowed = requirement.match != "definition" and any(
        need in requirement.evidence_needs for need in ("examples", "explanation")
    )
    unresolved_topic = False
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
            parameter_text = _complete_source_lines(evidence.element.text, max(start, section.start), min(end, section.end))
            needs = _section_needs(text, section, parameter_text)
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
            parameters = _described_parameters(parameter_text) if "parameters" in needs else {}
            parameter_names = (_declared_parameter_names(evidence.element.text, sections, section.owner)
                               if parameters and requirement.aspects else set())
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
                if _contains(text, aspect):
                    coverage.add(("aspect", target, aspect))
    if not coverage and not unresolved_topic:
        return None
    if requirement.version and version == "match":
        coverage.add(("version", "", requirement.version))
    relevance = len(matched_terms) / max(1, len(subjects)) if not requirement.symbols else 1.0
    return _Support(frozenset(coverage), relevance, confidence, version, unresolved_topic)


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
) -> DocsAssessment:
    """Assess only the supplied source ranges, without expanding or selecting them.

    Full source structure supplies ownership and section boundaries; only text
    inside each selection supplies factual coverage. This also checks the final
    cropped ranges sent to answer generation with the retrieval contract.
    """
    requirement = resolve_requirement(query, requirement)
    domains = library_domains(str(requirement.library or ""))
    if requirement.library and not domains:
        return DocsAssessment(hits=[], answerability="missing", missing=[f"library:{requirement.library}"])
    subjects = _topic_terms(query, str(requirement.library or ""))
    parsed = [(evidence, parse_doc_sections(evidence.element.text,
               title=evidence.snapshot.title, url=evidence.snapshot.source_uri))
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
        support = _evaluate_window(evidence, sections, requirement, subjects, focus, version)
        if support is not None:
            supports.append(support)
        elif requirement.symbols:
            rejected += 1
    return _assessment(supports, requirement, hits=[], rejected_symbols=rejected)


def select_docs_evidence(
    query: str, hits: list[SearchHit], requirement: RetrievalRequirement, *,
    k: int, allowed_domains: list[str], retrieval_warnings: list[str],
) -> DocsAssessment:
    """One scope/coverage model drives ranking, evidence selection and completion.

    Document formatting and provider scores cannot compensate for a different
    target. Only exact returned source ranges contribute to required coverage.
    """
    required = _required_coverage(requirement)
    subjects = _topic_terms(query, str(requirement.library or ""))
    parsed = [(hit, parse_doc_sections(hit.evidence.element.text,
                                      title=hit.evidence.snapshot.title, url=hit.evidence.snapshot.source_uri))
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
        evaluated = [_Candidate(window, support) for window in _candidate_windows(hit, sections, requirement)
                     if (support := _evaluate_window(window.evidence, sections, requirement, subjects, focus, version))]
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
