from __future__ import annotations

import re


def canonicalize_docs_query_text(text: str) -> str:
    """Collapse whitespace without interpreting punctuation in search prose."""
    return " ".join(text.split())


def canonicalize_docs_symbol_text(symbol: str) -> str:
    """Remove spacing around explicit path separators without changing identity."""
    return re.sub(r"\s*\.\s*", ".", symbol.strip(" `()"))
