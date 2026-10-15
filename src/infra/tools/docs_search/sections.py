"""Conservative document structure with offsets into the unchanged provider text.

Owners describe definitions or document identity, never APIs merely called in an
example. A missing owner is deliberately retained instead of guessed from a query.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class DocSection:
    owner: str
    kind: str
    start: int
    end: int
    confidence: int = 1


_NAME = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_NAME_PATTERN = re.compile(rf"^{_NAME}$")
_DECLARATION = re.compile(rf"^(?:(?:async\s+)?def\s+|class\s+|function\s+)?(?P<name>{_NAME})\s*\(")
_SECTION_NAMES = {
    "parameter": "parameters", "parameters": "parameters", "args": "parameters",
    "arguments": "parameters", "keyword arguments": "parameters",
    "return": "returns", "returns": "returns", "yields": "returns",
    "option": "options", "options": "options", "signature": "signature",
    "synopsis": "signature", "usage": "examples", "example": "examples",
    "examples": "examples", "note": "notes", "notes": "notes",
    "see also": "notes", "warning": "notes", "warnings": "notes",
    "raises": "notes", "references": "notes", "attributes": "notes",
    "methods": "notes", "description": "overview", "overview": "overview",
}
_GENERAL_NAMES = {
    "api", "reference", "documentation", "docs", "index", "manual", "guide",
    "tutorial", "tutorials", "contents", "introduction", "home", "navigation",
    "search", "source", "examples", "parameters", "returns", "notes", "options",
    "usage", "signature", "description", "overview", "attributes", "methods",
}


def clean_markup(value: str) -> str:
    """A recognition-only view; callers keep offsets from the original text."""
    value = re.sub(r"\[\[source\]\]\([^)]*\)", "", value, flags=re.I)
    value = re.sub(r"\[#\]\([^)]*\)", "", value)
    value = re.sub(r"\[([^]\n]+)\]\([^)]*\)", r"\1", value)
    # Comparison operators in prose/code are not tags. In particular, a '< 1'
    # must not erase later parameter entries up to an unrelated '> 1'.
    value = re.sub(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>\n]*)?\s*/?>", " ", value)
    value = html.unescape(value).replace("\\_", "_")
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1 ", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", value)
    return " ".join(value.replace("`", "").split()).strip()


def _symbol(value: str) -> str:
    cleaned = clean_markup(value).strip().strip("# ¶ ")
    cleaned = re.sub(r"^(?:class|function|method|def)\s+", "", cleaned)
    if _NAME_PATTERN.fullmatch(cleaned) and cleaned.casefold() not in _GENERAL_NAMES:
        return cleaned
    return ""


def _declaration(value: str) -> str:
    cleaned = clean_markup(value).strip().strip("# ¶ ")
    match = _DECLARATION.match(cleaned)
    if match is None or ")" not in cleaned:
        return ""
    # A prose sentence mentioning a call is not a declaration. Source links and
    # return annotations are allowed after a complete signature.
    tail = cleaned[cleaned.rfind(")") + 1:].strip()
    if tail and tail != ":" and not tail.startswith(("->", "→")):
        return ""
    return match.group("name")


def _sphinx_definition_owner(raw_line: str) -> str:
    """A standalone code declaration and its matching self-link identify a new API."""
    if raw_line[:1].isspace():
        return ""
    anchor = re.search(
        rf'\[(?:¶|#)\]\(#[ ]*(?P<name>{_NAME})[ ]+"(?:Permalink|Link) to this definition"\)\s*$',
        raw_line,
    )
    if anchor is None or "`" not in raw_line[:anchor.start()]:
        return ""
    declaration = _declaration(raw_line[:anchor.start()])
    target = anchor.group("name")
    return target if declaration and (target == declaration or target.endswith("." + declaration)) else ""


def resolve_document_symbol(*, title: str = "", url: str = "") -> str:
    """Return only an unambiguous identity supplied by the document itself."""
    title_part = re.split(r"\s+[—–|]\s+|\s+-\s+", str(title or ""), maxsplit=1)[0]
    title_part = re.sub(r"^\d+(?:\.\d+)*\.\s+", "", title_part)
    title_symbol = _symbol(title_part) or _declaration(title_part)
    parsed = urlparse(str(url or ""))
    leaf = unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1])
    leaf = re.sub(r"\.(?:html?|md|rst)$", "", leaf, flags=re.I)
    url_symbol = _symbol(leaf)
    if title_symbol:
        # A qualified URL may disambiguate a short title, but a different title
        # must not be silently replaced by another API from the URL.
        if url_symbol and "." not in title_symbol and url_symbol.endswith("." + title_symbol):
            return url_symbol
        return title_symbol
    stdlib_module = parsed.hostname == "docs.python.org" and "/library/" in parsed.path
    if url_symbol and (stdlib_module or "." in url_symbol or any(part in parsed.path.split("/") for part in ("api", "generated", "reference"))):
        return url_symbol
    return ""


def parse_doc_sections(text: str, *, title: str = "", url: str = "") -> list[DocSection]:
    """Partition source into owned structural sections without rewriting it."""
    if not text or not text.strip():
        return []
    identity = resolve_document_symbol(title=title, url=url)
    parsed_url = urlparse(url)
    module = (identity if parsed_url.hostname == "docs.python.org" and "/library/" in parsed_url.path
              and identity == unquote(parsed_url.path.rsplit("/", 1)[-1]).removesuffix(".html") else "")
    owner, confidence = identity, 1 if identity else 0
    kind, start, offset = "overview", 0, 0
    sections: list[DocSection] = []
    fence = ""
    before_fence: str | None = None
    owner_level: int | None = None
    lines = text.splitlines(keepends=True)

    def boundary(position: int, new_owner: str, new_kind: str, new_confidence: int) -> None:
        nonlocal start, owner, kind, confidence
        if position > start and text[start:position].strip():
            sections.append(DocSection(owner, kind, start, position, confidence))
        start, owner, kind, confidence = position, new_owner, new_kind, new_confidence

    def qualify(name: str) -> str:
        for known in (owner, identity):
            if name == known or name.startswith(known + "."):
                return name
            if known.endswith("." + name):
                return known
            # Relative class/member declarations inherit the source's namespace.
            # This never uses the requested query or creates API equivalences.
            parts = known.split(".")
            for index in range(1, len(parts)):
                relative = ".".join(parts[index:])
                if name.startswith(relative + "."):
                    return known + name[len(relative):]
        if module and ("." not in name or name.split(".", 1)[0][:1].isupper()):
            return module + "." + name
        return name

    for index, raw_line in enumerate(lines):
        line_start, offset = offset, offset + len(raw_line)
        stripped = raw_line.strip()
        fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence_match:
            marker = fence_match.group(1)
            if not fence:
                fence = marker
                first_line = next((lines[following].strip() for following in range(index + 1, len(lines))
                                   if lines[following].strip()), "")
                if kind not in {"examples", "signature"} or (kind == "signature" and first_line.startswith(">>>")):
                    before_fence = kind
                    boundary(line_start, owner, "examples", confidence)
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = ""
                if before_fence is not None:
                    boundary(offset, owner, before_fence, confidence)
                    before_fence = None
            continue
        if fence or not stripped:
            continue
        if stripped.startswith(">>>") and kind != "examples":
            boundary(line_start, owner, "examples", confidence)
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        html_heading = re.match(r"<h([1-6])\b[^>]*>(.*?)</h\1>", stripped, flags=re.I)
        setext = index + 1 < len(lines) and bool(re.fullmatch(r"\s*(?:={3,}|-{3,})\s*", lines[index + 1]))
        level = len(heading_match.group(1)) if heading_match else int(html_heading.group(1)) if html_heading else (1 if lines[index + 1].lstrip().startswith("=") else 2) if setext else None
        heading_text = heading_match.group(2) if heading_match else html_heading.group(2) if html_heading else stripped
        cleaned = clean_markup(heading_text).strip("# ¶ ")
        section_name = cleaned.rstrip(":").strip().casefold()
        new_kind = _SECTION_NAMES.get(section_name)
        if new_kind:
            boundary(line_start, owner, new_kind, confidence)
            continue
        if level is not None:
            heading_owner = _symbol(cleaned) or _declaration(cleaned)
            if heading_owner:
                owner_level = level
                heading_kind = "signature" if _declaration(cleaned) else "overview"
                boundary(line_start, qualify(heading_owner), heading_kind, 2)
            else:
                if owner_level is not None and level <= owner_level:
                    owner_level = None
                    boundary(line_start, identity, "overview", 1 if identity else 0)
                else:
                    boundary(line_start, owner, "overview", confidence)
            continue
        anchored_owner = _sphinx_definition_owner(raw_line)
        if anchored_owner:
            boundary(line_start, anchored_owner, "signature", 2)
            continue
        # API declarations belong in an overview/signature, not in example,
        # parameter, return, or note text that happens to contain a call.
        if kind in {"overview", "signature"}:
            declaration = _declaration(cleaned)
            if declaration:
                boundary(line_start, qualify(declaration), "signature", 2)
    if text[start:].strip():
        sections.append(DocSection(owner, kind, start, len(text), confidence))
    return sections


__all__ = ["DocSection", "parse_doc_sections", "resolve_document_symbol"]
