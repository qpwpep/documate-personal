from __future__ import annotations

import re
from collections.abc import Iterable

from src.core.contracts.routes import route_for_tool
from src.core.evidence import EvidenceItem


def _code_metadata_option_literals(item: EvidenceItem) -> list[str]:
    code_metadata = item.code_metadata if isinstance(item.code_metadata, dict) else {}
    options: list[str] = []
    seen: set[str] = set()
    for option in code_metadata.get("option_literals") or []:
        option_text = " ".join(str(option or "").split())
        compact = re.sub(r"\s+", "", option_text.lower())
        if not option_text or compact in seen:
            continue
        options.append(option_text)
        seen.add(compact)
    return options


def extract_uploaded_option_literals(evidence_items: Iterable[EvidenceItem]) -> list[str]:
    options: list[str] = []
    seen: set[str] = set()
    for item in evidence_items:
        if route_for_tool(str(item.tool or "")) != "upload":
            continue
        for option in _code_metadata_option_literals(item):
            compact = re.sub(r"\s+", "", option.lower())
            if compact and compact not in seen:
                options.append(option)
                seen.add(compact)
        for match in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^,\)\]\}\n]+", str(item.snippet or "")):
            option = " ".join(match.strip().split())
            compact = re.sub(r"\s+", "", option.lower())
            if compact and compact not in seen:
                options.append(option)
                seen.add(compact)
    return options


def contains_option_literal(text: str, options: list[str]) -> bool:
    compact_text = re.sub(r"\s+", "", str(text or "").lower())
    return any(re.sub(r"\s+", "", option.lower()) in compact_text for option in options)
