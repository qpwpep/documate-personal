from __future__ import annotations

import re
from src.core.planner_schema import RetrievalRequirement
from src.infra.tools.docs_search.normalization import canonicalize_docs_symbol_text
from src.infra.tools.docs_search.policy import docs_search_rules, infer_docs_library


_LIBRARY_NAMES = {"np": "numpy", "pd": "pandas", "plt": "matplotlib", "sklearn": "scikit-learn", "torch": "pytorch", "transformers": "Hugging Face", "huggingface": "Hugging Face", "bs4": "BeautifulSoup", "st": "streamlit"}
_SYMBOL_ALIASES = {"numpy": {"np": "numpy"}, "pandas": {"pd": "pandas"}, "matplotlib": {"plt": "matplotlib.pyplot"}, "pytorch": {"nn": "torch.nn"}, "streamlit": {"st": "streamlit"}, "scikit-learn": {"sklearn": "sklearn"}}


def canonical_symbol(symbol: str, library: str) -> str:
    value = canonicalize_docs_symbol_text(symbol)
    for alias, target in _SYMBOL_ALIASES.get(library.casefold(), {}).items():
        if value.startswith(alias + "."):
            return target + value[len(alias):]
    return value


def library_domains(library: str) -> list[str]:
    wanted = library.casefold()
    return list(dict.fromkeys(domain for hint in docs_search_rules().query_hints
                              if hint.library_name.casefold() == wanted for domain in hint.domains))


def resolve_requirement(requirement: RetrievalRequirement) -> RetrievalRequirement:
    """Keep canonical library IDs while resolving unscoped owners and symbol aliases."""
    library = requirement.library or ""
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
    # A search expression may contain literal code, quoted text or URL paths.
    value = query
    if requirement.version and requirement.version.casefold() not in value.casefold():
        value += " " + requirement.version
    return value
