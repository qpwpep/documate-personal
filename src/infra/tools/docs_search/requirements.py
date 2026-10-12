from __future__ import annotations

import re
from src.core.planner_schema import RetrievalRequirement
from src.infra.tools.docs_search.normalization import canonicalize_docs_query_text, canonicalize_docs_symbol_text
from src.infra.tools.docs_search.policy import docs_search_rules, infer_docs_library


_LIBRARY_NAMES = {"np": "numpy", "pd": "pandas", "plt": "matplotlib", "sklearn": "scikit-learn", "torch": "pytorch", "transformers": "Hugging Face", "huggingface": "Hugging Face", "bs4": "BeautifulSoup", "st": "streamlit"}
_SYMBOL_ALIASES = {"numpy": {"np": "numpy"}, "pandas": {"pd": "pandas"}, "matplotlib": {"plt": "matplotlib.pyplot"}, "pytorch": {"nn": "torch.nn"}, "streamlit": {"st": "streamlit"}, "scikit-learn": {"sklearn": "sklearn"}}


def canonical_library(value: str | None) -> str:
    text = str(value or "").strip()
    return _LIBRARY_NAMES.get(text.casefold(), text)


def canonical_symbol(symbol: str, library: str) -> str:
    value = canonicalize_docs_symbol_text(symbol)
    for alias, target in _SYMBOL_ALIASES.get(library.casefold(), {}).items():
        if value.startswith(alias + "."):
            return target + value[len(alias):]
    return value


def library_domains(library: str) -> list[str]:
    wanted = canonical_library(library).casefold()
    return list(dict.fromkeys(domain for hint in docs_search_rules().query_hints
                              if hint.library_name.casefold() == wanted for domain in hint.domains))


def resolve_requirement(requirement: RetrievalRequirement) -> RetrievalRequirement:
    """Canonicalize the fixed source requirement without deriving goals from query text."""
    library = canonical_library(requirement.library)
    if not library:
        scope_text = " ".join(requirement.symbols)
        prefixes = re.findall(r"(?<![\w.])([A-Za-z_]+)\s*\.", scope_text)
        library = next((_LIBRARY_NAMES[p.casefold()] for p in prefixes if p.casefold() in _LIBRARY_NAMES), "")
        library = library or infer_docs_library(scope_text) or ""
    return requirement.model_copy(update={
        "library": library or None,
        "symbols": [canonical_symbol(symbol, library) for symbol in requirement.symbols],
    })


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
