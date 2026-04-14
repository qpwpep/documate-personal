from __future__ import annotations

import re


_LEADING_TITLE_PATTERN = re.compile(r"(?i)^title:\s*")
_MARKDOWN_LINK_PATTERN = re.compile(r"!?\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_MARKDOWN_HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s+")
_LEADING_MARKDOWN_PATTERN = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)")
_MARKDOWN_DECORATION_PATTERN = re.compile(r"[*~`]+")
_DOC_SECTION_HEADING_PATTERN = re.compile(
    r"(?i)^(?:parameters?|returns?|examples?|notes?|see also|references?|attributes?|methods?)$"
)
_DOC_REFERENCE_TITLE_PATTERN = re.compile(
    r"(?i)\b(?:documentation|docs?|reference|api reference|user guide)\b"
)
_NAVIGATION_LINE_PATTERNS = (
    re.compile(r"(?i)^(?:api|api reference|documentation|docs?|guide|reference|tutorials?|user guide)$"),
    re.compile(r"(?i)^(?:table of contents|contents|on this page|in this article)$"),
    re.compile(r"(?i)^(?:next|previous|prev|back to top|edit this page|view source|search|skip to content)$"),
    re.compile(r"(?i)^(?:navigation|menu|breadcrumbs?|home)$"),
)
_NAVIGATION_PREFIX_PATTERNS = (
    re.compile(r"(?i)^(?:table of contents|contents|on this page|in this article)\b"),
    re.compile(r"(?i)^(?:next|previous|prev)\s*[:\-]?\s+\S"),
    re.compile(r"(?i)^(?:navigation|menu|breadcrumbs?)\s*[:\-]?\s+\S"),
)
_NAVIGATION_EMBEDDED_PATTERNS = (
    re.compile(r"(?i)\bskip to content\b"),
    re.compile(r"(?i)\bon this page\b"),
    re.compile(r"(?i)\btable of contents\b"),
    re.compile(r"(?i)\bedit this page\b"),
    re.compile(r"(?i)\bview source\b"),
)
_BREADCRUMB_SPLIT_PATTERN = re.compile(r"\s*(?:[>]|[|]|/|→)\s*")
_BREADCRUMB_WORDS = {
    "api",
    "article",
    "back",
    "content",
    "reference",
    "references",
    "doc",
    "docs",
    "documentation",
    "edit",
    "guide",
    "guides",
    "home",
    "in",
    "learn",
    "navigation",
    "next",
    "of",
    "on",
    "overview",
    "page",
    "previous",
    "search",
    "skip",
    "source",
    "this",
    "to",
    "tutorial",
    "tutorials",
    "user",
    "view",
}
_DOC_SECTION_WORDS = {
    "parameters",
    "parameter",
    "returns",
    "return",
    "examples",
    "example",
    "notes",
    "note",
    "references",
    "reference",
    "attributes",
    "attribute",
    "methods",
    "method",
    "see",
    "also",
}
_PLAIN_LANGUAGE_SIGNATURE_PREFIXES = {
    "allow",
    "allows",
    "call",
    "calls",
    "create",
    "creates",
    "join",
    "joins",
    "pass",
    "passes",
    "return",
    "returns",
    "set",
    "sets",
    "split",
    "splits",
    "use",
    "uses",
}


def _normalize_doc_line(line: str) -> str:
    normalized = str(line or "").strip()
    if not normalized:
        return ""
    normalized = re.sub(r"\\([_*#`])", r"\1", normalized)
    normalized = normalized.replace("\\", "")
    normalized = re.sub(r"\s*[#]+\s*", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" -|:")


def _looks_like_navigation_line(line: str) -> bool:
    if any(pattern.fullmatch(line) for pattern in _NAVIGATION_LINE_PATTERNS):
        return True
    if any(pattern.match(line) for pattern in _NAVIGATION_PREFIX_PATTERNS):
        return True

    line_words = {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9-]*", line)}
    if line_words and len(line_words) <= 8 and line_words.issubset(_BREADCRUMB_WORDS):
        return True

    breadcrumb_segments = [
        segment.strip()
        for segment in _BREADCRUMB_SPLIT_PATTERN.split(line)
        if segment.strip()
    ]
    if len(breadcrumb_segments) <= 1:
        return False
    normalized_segments = [re.sub(r"\s+", " ", segment).strip().lower() for segment in breadcrumb_segments]
    if any(len(segment.split()) > 4 for segment in normalized_segments):
        return False
    if normalized_segments[0] == "home":
        return True
    if any(segment in _BREADCRUMB_WORDS for segment in normalized_segments[:-1]):
        return True
    return False


def _looks_like_signature_line(line: str) -> bool:
    if len(line) > 160 or "(" not in line:
        return False
    prefix, _, suffix = line.partition("(")
    identifier = prefix.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{1,}", identifier):
        return False
    inner = suffix.rsplit(")", 1)[0] if ")" in suffix else suffix
    return bool(inner.strip()) and any(marker in inner for marker in (",", "=", "*", "[", "]"))


def _looks_like_signature_fragment(line: str) -> bool:
    normalized = _normalize_doc_line(line)
    if len(normalized) > 220 or "(" not in normalized:
        return False
    prefix, _, suffix = normalized.partition("(")
    prefix_tokens = [token for token in re.sub(r"[#:.]", " ", prefix).split() if token]
    if not prefix_tokens or len(prefix_tokens) > 6:
        return False
    if prefix_tokens[0].lower() in _PLAIN_LANGUAGE_SIGNATURE_PREFIXES:
        return False
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", token) for token in prefix_tokens):
        return False
    inner = suffix.rsplit(")", 1)[0] if ")" in suffix else suffix
    return bool(inner.strip()) and any(marker in inner for marker in (",", "=", "*", "[", "]"))


def _looks_like_title_only_line(line: str) -> bool:
    if len(line) > 60 or any(punct in line for punct in ".!?"):
        return False
    if "(" in line or ")" in line:
        return False
    words = line.split()
    if not 1 <= len(words) <= 4:
        return False
    return all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", word) for word in words)


def _looks_like_identifier_only_fragment(line: str) -> bool:
    stripped = str(line or "").strip().rstrip(".!?")
    words = stripped.split()
    if not words or len(words) > 3:
        return False
    return all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", word) for word in words)


def _looks_like_section_listing(line: str) -> bool:
    section_words = [word.lower() for word in re.findall(r"[A-Za-z]+", line)]
    return bool(section_words) and 2 <= len(section_words) <= 8 and all(
        word in _DOC_SECTION_WORDS for word in section_words
    )


def _looks_like_doc_chrome_line(line: str) -> bool:
    normalized = _normalize_doc_line(line)
    if not normalized:
        return False
    if _DOC_SECTION_HEADING_PATTERN.fullmatch(normalized):
        return True
    if _looks_like_section_listing(normalized):
        return True
    if _looks_like_signature_line(normalized):
        return True
    if _looks_like_signature_fragment(normalized):
        return True
    if len(normalized.split()) <= 12 and _DOC_REFERENCE_TITLE_PATTERN.search(normalized):
        return True
    return _looks_like_title_only_line(normalized)


def _ensure_sentence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


def clean_grounded_text(text: str) -> str:
    cleaned = _LEADING_TITLE_PATTERN.sub("", str(text or "").strip())
    if not cleaned:
        return ""

    filtered_lines: list[str] = []
    for raw_line in cleaned.replace("\r", "\n").split("\n"):
        is_markdown_heading = _MARKDOWN_HEADING_PATTERN.match(raw_line) is not None
        is_markdown_link_only = _MARKDOWN_LINK_PATTERN.fullmatch(raw_line.strip()) is not None
        line = _MARKDOWN_IMAGE_PATTERN.sub(" ", raw_line)
        for pattern in _NAVIGATION_EMBEDDED_PATTERNS:
            line = pattern.sub(" ", line)
        line = _MARKDOWN_LINK_PATTERN.sub(r"\1", line)
        line = _HTML_TAG_PATTERN.sub(" ", line)
        line = _LEADING_MARKDOWN_PATTERN.sub("", line).strip()
        line = _MARKDOWN_DECORATION_PATTERN.sub(" ", line)
        line = _normalize_doc_line(line)
        if not line:
            continue
        if is_markdown_heading and len(line.split()) <= 8:
            continue
        if is_markdown_link_only and len(line.split()) <= 8:
            continue
        if _looks_like_navigation_line(line) or _looks_like_doc_chrome_line(line):
            continue
        if not filtered_lines and _looks_like_title_only_line(line):
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{1,}\([^)]*\)", line):
            continue
        filtered_lines.append(line)

    cleaned = " ".join(filtered_lines)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+\.\.\.\s*$", "", cleaned)
    if _looks_like_doc_chrome_line(cleaned):
        return ""
    return cleaned.strip()


def summarize_grounded_text(text: str, *, max_chars: int = 220) -> str:
    cleaned = clean_grounded_text(text)
    if not cleaned:
        return ""

    first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    summary = first_sentence or cleaned
    if len(summary) > max_chars:
        summary = (summary[:max_chars].rsplit(" ", 1)[0] or summary[:max_chars]).rstrip(" ,;:")
    if _looks_like_doc_chrome_line(summary) or _looks_like_signature_fragment(summary):
        return ""
    if _looks_like_identifier_only_fragment(summary):
        return ""
    return _ensure_sentence(summary)


__all__ = [
    "clean_grounded_text",
    "summarize_grounded_text",
]
