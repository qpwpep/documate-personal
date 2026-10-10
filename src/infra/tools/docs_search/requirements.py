from __future__ import annotations

import re
from src.core.planner_schema import DocsEvidenceNeed, RetrievalRequirement
from src.infra.tools.docs_search.normalization import (
    canonicalize_docs_query_text, normalize_identifier_reference_text, normalize_identifier_token,
)
from src.infra.tools.docs_search.policy import docs_search_rules, infer_docs_library


_ASCII_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9._-]*|v\d+)(?![A-Za-z0-9_])"
)


def _identifier_stopwords(*, library_name: str = "") -> set[str]:
    stopwords = {item.lower() for item in docs_search_rules().docs_identifier_stopwords}
    stopwords.update(
        {
            "api",
            "parameters",
            "parameter",
            "reference",
            "validation",
            "validator",
        }
    )
    for part in re.findall(r"[A-Za-z0-9_.-]+", str(library_name or "").lower()):
        stopwords.add(part)
    return stopwords


def extract_exact_identifier_terms(query: str, *, library_name: str = "") -> list[str]:
    identifiers: list[str] = []
    seen: set[str] = set()
    stopwords = _identifier_stopwords(library_name=library_name)
    normalized_query = normalize_identifier_reference_text(query)
    for token in _ASCII_IDENTIFIER_PATTERN.findall(normalized_query):
        normalized = normalize_identifier_token(token)
        lowered = normalized.lower()
        if not normalized or lowered in stopwords:
            continue
        if re.fullmatch(r"v\d+", lowered):
            continue
        if not (
            "." in normalized
            or "_" in normalized
            or "-" in normalized
            or normalized != normalized.lower()
        ):
            continue
        if lowered not in seen:
            identifiers.append(normalized)
            seen.add(lowered)
    return identifiers


_LIBRARY_NAMES = {"np": "numpy", "pd": "pandas", "plt": "matplotlib", "sklearn": "scikit-learn", "torch": "pytorch", "transformers": "Hugging Face", "huggingface": "Hugging Face", "bs4": "BeautifulSoup", "st": "streamlit"}
_SYMBOL_ALIASES = {"numpy": {"np": "numpy"}, "pandas": {"pd": "pandas"}, "matplotlib": {"plt": "matplotlib.pyplot"}, "pytorch": {"nn": "torch.nn"}, "streamlit": {"st": "streamlit"}, "scikit-learn": {"sklearn": "sklearn"}}


def canonical_library(value: str | None) -> str:
    text = str(value or "").strip()
    return _LIBRARY_NAMES.get(text.casefold(), text)


def canonical_symbol(symbol: str, library: str) -> str:
    value = canonicalize_docs_query_text(symbol).strip(" `()")
    for alias, target in _SYMBOL_ALIASES.get(library.casefold(), {}).items():
        if value.startswith(alias + "."):
            return target + value[len(alias):]
    return value


def library_domains(library: str) -> list[str]:
    wanted = canonical_library(library).casefold()
    return list(dict.fromkeys(domain for hint in docs_search_rules().query_hints
                              if hint.library_name.casefold() == wanted for domain in hint.domains))


def resolve_requirement(query: str, requirement: RetrievalRequirement | dict | None) -> RetrievalRequirement:
    """Normalize fixed requirements; infer goals only for an unplanned direct call."""
    supplied = RetrievalRequirement.model_validate(requirement) if requirement is not None else RetrievalRequirement()
    library = canonical_library(supplied.library)
    if not library:
        # Namespace aliases identify libraries before library-specific normalization.
        # A fixed contract can resolve its own names, never new query vocabulary.
        scope_text = " ".join(supplied.symbols) if requirement is not None else query
        prefixes = re.findall(r"(?<![\w.])([A-Za-z_]+)\s*\.", scope_text)
        library = next((_LIBRARY_NAMES[p.casefold()] for p in prefixes if p.casefold() in _LIBRARY_NAMES), "")
        library = library or infer_docs_library(scope_text) or ""
    if requirement is not None:
        return supplied.model_copy(update={
            "library": library or None,
            "symbols": [canonical_symbol(s, library) for s in supplied.symbols],
        })
    symbols = extract_exact_identifier_terms(query, library_name=library)
    return RetrievalRequirement(
        library=library or None,
        symbols=[canonical_symbol(s, library) for s in symbols],
        evidence_needs=infer_evidence_needs(query),
        match="symbol" if symbols else "topic",
    )


def infer_evidence_needs(query: str) -> list[DocsEvidenceNeed]:
    """Infer information kinds, never invent parameter names or allowed values."""
    patterns: dict[DocsEvidenceNeed, str] = {
        "parameters": r"\b(?:parameters?|params?|arguments?|args?|options?)\b|옵션|파라미터|매개변수|인자",
        "signature": r"\bsignature\b|시그니처|호출\s*문법",
        "examples": r"\bexamples?\b|예제|사용\s*예시|코드\s*예시",
    }
    return sorted(need for need, pattern in patterns.items() if re.search(pattern, query, re.I))


def canonical_query(query: str, requirement: RetrievalRequirement) -> str:
    value = canonicalize_docs_query_text(query)
    for alias, target in _SYMBOL_ALIASES.get(str(requirement.library or "").casefold(), {}).items():
        value = re.sub(rf"(?<![\w.]){re.escape(alias)}\.", target + ".", value)
    if requirement.version and requirement.version.casefold() not in value.casefold():
        value += " " + requirement.version
    return value


def reformulate_query(query: str, requirement: RetrievalRequirement, missing: list[str]) -> str:
    # Strengthen scope without replacing prose constraints or adding a new topic.
    refined = query
    for term in dict.fromkeys([*requirement.symbols, str(requirement.version or "")]):
        if term:
            refined = _quote_search_term(refined, term)
    for aspect in requirement.aspects:
        if aspect.casefold() not in refined.casefold():
            refined += " " + aspect
    needs = [need for need in requirement.evidence_needs
             if any(item.startswith(f"need:{need}") for item in missing)]
    qualifier = "examples" if needs == ["examples"] else "API reference"
    if not re.search(rf"\b{qualifier}\b", refined, flags=re.I):
        refined += " " + qualifier
    for need in needs:
        if not re.search(rf"\b{need}\b", refined, flags=re.I):
            refined += " " + need
    return refined


def _quote_search_term(query: str, term: str) -> str:
    pattern = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(term)}(?![A-Za-z0-9_.])")
    # Existing quoted phrases already constrain their contents; never nest quotes.
    pieces = re.split(r'("(?:\\.|[^"\\])*")', query)
    found = False
    for index, piece in enumerate(pieces):
        if pattern.search(piece) is None:
            continue
        found = True
        if index % 2 == 0:
            pieces[index] = pattern.sub(lambda match: f'"{match.group()}"', piece)
    refined = "".join(pieces)
    return refined if found else f'{refined} "{term}"'
