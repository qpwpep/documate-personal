"""Literal source anchors with code-sensitive identifier matching."""

from __future__ import annotations

import re

from src.core.evidence import EvidenceRef


def _contains_literal(anchor: str, excerpt: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(anchor)}(?!\w)", excerpt) is not None


def literal_aspect_present(aspect: str, text: str, *, case_sensitive: bool) -> bool:
    """Collapse whitespace for matching only; preserve case for code source ranges."""
    anchor = " ".join(aspect.split())
    excerpt = " ".join(text.split())
    if not case_sensitive:
        anchor, excerpt = anchor.casefold(), excerpt.casefold()
    return _contains_literal(anchor, excerpt)


def missing_literal_aspects(aspects: list[str], evidence: list[EvidenceRef]) -> list[str]:
    """Report absent anchors without inferring semantic support or changing source text."""
    excerpts = [(" ".join(item.excerpt.split()), item.element.kind == "code") for item in evidence]
    excerpts = [(text if is_code else text.casefold(), is_code) for text, is_code in excerpts]
    missing = []
    for aspect in aspects:
        anchor = " ".join(aspect.split())
        folded_anchor = anchor.casefold()
        if not any(_contains_literal(anchor if is_code else folded_anchor, text) for text, is_code in excerpts):
            missing.append(aspect)
    return missing
