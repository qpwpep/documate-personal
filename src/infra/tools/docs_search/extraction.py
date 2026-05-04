from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import urlparse


_DETAIL_QUERY_MARKERS = (
    "api",
    "reference",
    "parameter",
    "parameters",
    "argument",
    "arguments",
    "option",
    "options",
    "signature",
    "method",
    "function",
    "옵션",
    "파라미터",
    "매개변수",
    "인자",
    "class",
    "사용법",
    "문법",
    "파라미터",
    "매개변수",
    "옵션",
    "인자",
    "정리",
    "예제",
)
_SECTION_ALIASES = {
    "parameters": "parameters",
    "parameter": "parameters",
    "args": "parameters",
    "arguments": "parameters",
    "returns": "returns",
    "return": "returns",
    "yields": "returns",
    "options": "options",
    "option": "options",
    "synopsis": "signature",
    "usage": "signature",
    "examples": "examples",
    "example": "examples",
    "notes": "notes",
    "note": "notes",
    "description": "overview",
}
_DOC_FAMILY_BY_DOMAIN = {
    "docs.python.org": "sphinx",
    "matplotlib.org": "sphinx_api",
    "numpy.org": "sphinx_api",
    "pandas.pydata.org": "pydata_sphinx",
    "docs.pytorch.org": "sphinx_api",
    "scikit-learn.org": "sphinx_api",
    "git-scm.com": "git_manpage",
    "fastapi.tiangolo.com": "mkdocs",
    "docs.pydantic.dev": "mkdocs",
    "crummy.com": "longpage",
    "python.langchain.com": "mdx",
    "huggingface.co": "mdx",
    "docs.streamlit.io": "mdx",
    "gradio.app": "mdx",
}
_PARAMETER_STOPWORDS = {
    "a",
    "an",
    "and",
    "added",
    "default",
    "for",
    "if",
    "in",
    "it",
    "note",
    "see",
    "the",
    "this",
    "to",
    "when",
}
_TAG_BLOCK_PATTERN = re.compile(r"</?(?:p|br|div|section|article|main|li|dt|dd|tr|th|td|h[1-6]|pre|code)[^>]*>", re.I)
_SCRIPT_STYLE_PATTERN = re.compile(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", re.I | re.S)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_DEFAULT_PATTERN = re.compile(r"\bdefault\s*[:=]\s*([^,;.]+)", re.I)
_IDENTIFIER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_.-]*|\*{1,2}[A-Za-z_][A-Za-z0-9_]*)")


def should_extract_doc_content(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(marker in lowered for marker in _DETAIL_QUERY_MARKERS)


def extract_doc_content(
    *,
    url: str,
    title: Any,
    content: Any,
    query: str,
) -> tuple[dict[str, Any] | None, str | None]:
    text = _normalize_content(content)
    if not text:
        return None, None

    lines = _clean_lines(text)
    if not lines:
        return None, None

    doc_family = _doc_family_for_url(url)
    symbol = _resolve_symbol(title=title, url=url, query=query)
    sections = _split_sections(lines)
    source_sections = [section for section, section_lines in sections.items() if section_lines]

    signature = _find_signature(lines=lines, sections=sections, symbol=symbol, doc_family=doc_family)
    parameters = _extract_entries(sections.get("parameters", []), mode="parameters")
    returns = _extract_entries(sections.get("returns", []), mode="returns")
    options = _extract_entries(sections.get("options", []), mode="options")
    if doc_family == "git_manpage" and not options:
        options = _extract_entries(lines, mode="options")

    examples = _extract_text_samples(sections.get("examples", []), limit=3)
    notes = _extract_text_samples(sections.get("notes", []), limit=3)

    metadata = _drop_empty(
        {
            "doc_family": doc_family,
            "symbol": symbol,
            "signature": signature,
            "parameters": parameters,
            "returns": returns,
            "options": options,
            "examples": examples,
            "notes": notes,
            "source_sections": source_sections,
        }
    )
    if not metadata:
        return None, None

    snippet = _build_structured_snippet(metadata)
    return metadata, snippet


def _normalize_content(value: Any) -> str:
    text = str(value or "").replace("\r", "\n").strip()
    if not text:
        return ""
    if "<" in text and ">" in text:
        text = _SCRIPT_STYLE_PATTERN.sub("\n", text)
        text = _TAG_BLOCK_PATTERN.sub("\n", text)
        text = _TAG_PATTERN.sub(" ", text)
    return html.unescape(text)


def _clean_lines(text: str) -> list[str]:
    cleaned: list[str] = []
    for raw_line in text.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue
        lowered = line.lower()
        if lowered in {"skip to content", "back to top", "on this page", "search", "navigation"}:
            continue
        if lowered.startswith(("previous:", "next:", "edit this page", "view source")):
            continue
        cleaned.append(line)
    return cleaned


def _normalize_line(line: str) -> str:
    stripped = " ".join(str(line or "").replace("\xa0", " ").split()).strip()
    stripped = re.sub(r"^#{1,6}\s*", "", stripped)
    stripped = re.sub(r"^\s*[-*]\s+", "", stripped)
    stripped = stripped.strip("` ")
    stripped = stripped.replace("\\_", "_")
    stripped = re.sub(r"\*\*([^*]+)\*\*", r"\1", stripped)
    return stripped.strip()


def _doc_family_for_url(url: str) -> str:
    parsed = urlparse(str(url or ""))
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return _DOC_FAMILY_BY_DOMAIN.get(domain, "generic_docs")


def _resolve_symbol(*, title: Any, url: str, query: str) -> str:
    candidates = [
        str(title or ""),
        urlparse(str(url or "")).path.rsplit("/", 1)[-1].removesuffix(".html"),
        str(query or ""),
    ]
    for candidate in candidates:
        cleaned = re.split(r"\s+[—|-]\s+|\s+\|\s+", candidate.strip(), maxsplit=1)[0]
        match = re.search(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", cleaned)
        if match:
            return match.group(0)
    for candidate in candidates:
        match = re.search(r"[A-Za-z_][A-Za-z0-9_-]{2,}", candidate)
        if match:
            return match.group(0).replace("-", " ")
    return ""


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"overview": []}
    current = "overview"
    for line in lines:
        heading = _canonical_section_heading(line)
        if heading:
            current = heading
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _canonical_section_heading(line: str) -> str:
    normalized = line.strip().strip(":").lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in _SECTION_ALIASES:
        return _SECTION_ALIASES[normalized]
    if normalized.startswith("parameters:"):
        return "parameters"
    if normalized.startswith("returns:"):
        return "returns"
    if normalized.startswith("options:"):
        return "options"
    if normalized.startswith("examples:"):
        return "examples"
    if normalized.startswith("notes:"):
        return "notes"
    return ""


def _find_signature(
    *,
    lines: list[str],
    sections: dict[str, list[str]],
    symbol: str,
    doc_family: str,
) -> str:
    if doc_family == "git_manpage":
        synopsis_lines = sections.get("signature", [])
        if synopsis_lines:
            return _truncate(" ".join(synopsis_lines[:3]), 420)
    for line in lines:
        if "(" not in line or ")" not in line:
            continue
        if len(line) > 520:
            continue
        if symbol and symbol.lower() not in line.lower():
            continue
        return line
    for line in lines:
        if "(" in line and ")" in line and len(line) <= 420:
            return line
    return ""


def _extract_entries(lines: list[str], *, mode: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    descriptions: list[str] = []

    def flush() -> None:
        nonlocal current, descriptions
        if current is None:
            return
        description = " ".join(descriptions).strip()
        if description and not current.get("description"):
            current["description"] = _truncate(description, 360)
        entries.append({key: value for key, value in current.items() if value})
        current = None
        descriptions = []

    for line in lines:
        parsed = _parse_entry_header(line, mode=mode)
        if parsed is not None:
            flush()
            current = parsed
            continue
        if current is not None and line:
            descriptions.append(line)
    flush()
    return entries[:16]


def _parse_entry_header(line: str, *, mode: str) -> dict[str, str] | None:
    candidate = _normalize_line(line)
    if not candidate or len(candidate) > 180:
        return None
    if mode == "options" and candidate.startswith("-"):
        return _parse_option_header(candidate)
    if candidate.endswith(".") and "default" not in candidate.lower():
        return None

    candidate = candidate.replace(" : ", " ")
    parts = candidate.split(maxsplit=1)
    if not parts:
        return None
    name = parts[0].strip("`*,")
    if not _is_doc_entry_name(name):
        return None
    lowered_name = name.lower().strip("-")
    if lowered_name in _PARAMETER_STOPWORDS:
        return None
    type_text = parts[1].strip(" -:") if len(parts) > 1 else ""
    if not type_text and mode != "returns":
        return None
    default = _extract_default(type_text)
    return {
        "name": name,
        "type": _truncate(type_text, 120),
        "default": default,
    }


def _parse_option_header(candidate: str) -> dict[str, str] | None:
    match = re.match(r"(?P<name>--?[A-Za-z0-9][A-Za-z0-9-]*)(?:[,\s]+(?P<rest>.*))?$", candidate)
    if match is None:
        return None
    return {
        "name": match.group("name"),
        "type": _truncate(match.group("rest") or "", 120),
    }


def _is_doc_entry_name(value: str) -> bool:
    if not value:
        return False
    if value.startswith("-"):
        return True
    return _IDENTIFIER_PATTERN.fullmatch(value) is not None


def _extract_default(text: str) -> str:
    match = _DEFAULT_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def _extract_text_samples(lines: list[str], *, limit: int) -> list[str]:
    samples: list[str] = []
    in_code_block = False
    code_lines: list[str] = []
    for line in lines:
        if line.startswith("```"):
            if in_code_block:
                block = "\n".join(code_lines).strip()
                if block:
                    samples.append(_truncate(block, 500))
                code_lines = []
            in_code_block = not in_code_block
            continue
        if in_code_block:
            code_lines.append(line)
            continue
        if line:
            samples.append(_truncate(line, 260))
        if len(samples) >= limit:
            break
    return samples[:limit]


def _build_structured_snippet(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    signature = str(metadata.get("signature") or "").strip()
    if signature:
        parts.append(f"signature: {signature}")
    for key, label in (("parameters", "param"), ("options", "option"), ("returns", "return")):
        for entry in metadata.get(key) or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            detail = str(entry.get("description") or entry.get("type") or "").strip()
            if not name:
                continue
            parts.append(f"{label} {name}: {_truncate(detail, 140)}".strip())
            if len(parts) >= 12:
                return "\n".join(parts)
    for note in metadata.get("notes") or []:
        parts.append(f"note: {_truncate(str(note), 160)}")
        if len(parts) >= 12:
            break
    return "\n".join(parts)


def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in ("", None, []) and value != {}
    }


def _truncate(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


__all__ = ["extract_doc_content", "should_extract_doc_content"]
