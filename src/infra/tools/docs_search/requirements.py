from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

from src.core.evidence import SearchHit, build_evidence, dedupe_search_hits
from src.core.planner_schema import RetrievalRequirement
from src.infra.tools.docs_search.normalization import canonicalize_docs_query_text
from src.infra.tools.docs_search.policy import docs_search_rules, infer_docs_query_hint, result_matches_domains
from src.infra.tools.docs_search.ranking import extract_exact_identifier_terms, hit_has_grounded_text


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
    supplied = RetrievalRequirement.model_validate(requirement or {})
    if supplied.library or supplied.symbols or supplied.version or supplied.aspects:
        library = canonical_library(supplied.library)
        if not library and (hint := infer_docs_query_hint(query)):
            library = hint[0]
        return supplied.model_copy(update={"library": library or None,
                                           "symbols": [canonical_symbol(s, library) for s in supplied.symbols]})
    hint = infer_docs_query_hint(query)
    library = hint[0] if hint else ""
    symbols = [canonical_symbol(s, library) for s in extract_exact_identifier_terms(query, library_name=library)]
    return RetrievalRequirement(library=library or None, symbols=symbols,
                                match="symbol" if symbols else "topic")


def canonical_query(query: str, requirement: RetrievalRequirement) -> str:
    value = canonicalize_docs_query_text(query)
    for alias, target in _SYMBOL_ALIASES.get(str(requirement.library or "").casefold(), {}).items():
        value = re.sub(rf"(?<![\w.]){re.escape(alias)}\.", target + ".", value)
    if requirement.version and requirement.version.casefold() not in value.casefold():
        value += " " + requirement.version
    return value


def reformulate_query(query: str, requirement: RetrievalRequirement) -> str:
    # Strengthen scope without replacing prose constraints or adding a new topic.
    refined = query
    for term in dict.fromkeys([*requirement.symbols, str(requirement.version or "")]):
        if term:
            refined = _quote_search_term(refined, term)
    for aspect in requirement.aspects:
        if aspect.casefold() not in refined.casefold():
            refined += " " + aspect
    if re.search(r"\bapi\s+reference\b", refined, flags=re.I) is None:
        refined += " API reference"
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


def _contains(text: str, term: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text, re.I) is not None


def _symbol_occurs(text: str, symbol: str) -> bool:
    # A qualified symbol cannot match a child member or another namespace. An
    # unqualified symbol may match the leaf of a qualified declaration.
    preceding = r"[A-Za-z0-9_.]" if "." in symbol else r"[A-Za-z0-9_]"
    return re.search(rf"(?<!{preceding}){re.escape(symbol)}(?![A-Za-z0-9_.])", text) is not None


def _defines_symbol(hit: SearchHit, symbol: str) -> bool:
    """A title, API URL or definition line owns a symbol; a prose mention does not."""
    snapshot = hit.evidence.snapshot
    title = snapshot.title
    parsed = urlparse(snapshot.source_uri)
    reference_path = re.sub(r"\.(?:html?|md)$", "", unquote(parsed.path), flags=re.I)
    if _symbol_occurs(title, symbol) or _symbol_occurs(reference_path + "#" + unquote(parsed.fragment), symbol):
        return True
    for line in hit.evidence.element.text.splitlines():
        stripped = line.strip()
        if re.match(rf"^#{{1,6}}\s+(?:class\s+)?`?{re.escape(symbol)}(?:[\s`(#]|$)", stripped):
            return True
        if re.match(rf"^(?:class\s+|def\s+)`?{re.escape(symbol)}\s*\(", stripped):
            return True
    return False


def _matches_version(hit: SearchHit, version: str | None) -> bool:
    if not version:
        return True
    identity = f"{urlparse(hit.evidence.snapshot.source_uri).path} {hit.evidence.snapshot.title}"
    wanted = version.strip().removeprefix("v")
    if wanted.lower() in {"latest", "stable", "current"}:
        return re.search(r"/(?:latest|stable)/", identity, re.I) is not None
    return re.search(rf"(?<![0-9.])v?{re.escape(wanted)}(?:\.\d+)*(?![0-9.])", identity, re.I) is not None


def _aspect_present(text: str, aspect: str) -> bool:
    aliases = {"signature": r"\w\s*\([^\n]*\)", "parameters": r"\b(?:parameters|arguments|args)\b", "examples": r"\bexamples?\b|>>>"}
    if aspect.casefold() in aliases:
        return re.search(aliases[aspect.casefold()], text, re.I) is not None
    return _contains(text, aspect)


def _source_windows(hit: SearchHit, aspects: list[str]) -> list[SearchHit]:
    selected = [hit]
    text = hit.evidence.element.text
    for aspect in aspects:
        if any(_aspect_present(item.evidence.excerpt, aspect) for item in selected):
            continue
        match = re.search(re.escape(aspect), text, re.I)
        if match is None:
            continue
        start = max(0, text.rfind("\n", 0, match.start()))
        end = min(len(text), start + 1600)
        evidence = build_evidence(snapshot=hit.evidence.snapshot, element=hit.evidence.element, start=start, end=end)
        selected.append(hit.model_copy(update={"evidence": evidence}))
    return selected


def assess_candidates(hits: list[SearchHit], requirement: RetrievalRequirement, *, k: int) -> tuple[list[SearchHit], str, list[str]]:
    domains = set(library_domains(str(requirement.library or "")))
    scoped = [hit for hit in hits if hit_has_grounded_text(hit)
              and (not domains or result_matches_domains(hit.evidence.snapshot.source_uri, domains))
              and _matches_version(hit, requirement.version)]
    symbols = requirement.symbols
    eligible = [hit for hit in scoped if not symbols or any(_defines_symbol(hit, symbol) for symbol in symbols)]
    if not symbols and requirement.aspects:
        eligible = [hit for hit in eligible if any(_aspect_present(hit.evidence.element.text, aspect)
                                                  for aspect in requirement.aspects)]
    ordered: list[SearchHit] = []
    remaining = set(symbols)
    pending = list(eligible)
    while pending:
        best = max(pending, key=lambda h: sum(_defines_symbol(h, s) for s in remaining))
        pending.remove(best)
        ordered.append(best)
        remaining = {symbol for symbol in remaining if not _defines_symbol(best, symbol)}
    selected = dedupe_search_hits([item for hit in ordered for item in _source_windows(hit, requirement.aspects)])[:k]
    missing = [f"symbol:{symbol}" for symbol in symbols if not any(_defines_symbol(h, symbol) for h in selected)]
    if requirement.version and not scoped:
        missing.append(f"version:{requirement.version}")
    text = "\n".join(h.evidence.excerpt for h in selected)
    missing.extend(f"aspect:{a}" for a in requirement.aspects if not _aspect_present(text, a))
    if not selected and not missing:
        missing.append("topic")
    return selected, ("partial" if missing and selected else "missing" if missing else "covered"), missing
