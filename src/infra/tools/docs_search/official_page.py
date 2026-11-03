"""Capture bounded official pages for the requested API and version."""

from __future__ import annotations

from html.parser import HTMLParser
from itertools import groupby
import re
import zlib
from urllib.parse import urldefrag, urljoin, urlparse

import requests

from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import EvidenceRef, RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import RetrievalRequirement
from src.infra.llm_boundary import bounded_request_timeout, check_request_deadline
from src.infra.tools.docs_search.policy import (
    is_allowed_doc_url, is_current_stable_doc_url, normalized_domain_set, result_matches_domains,
)


_MAX_PAGE_BYTES = 2 * 1024 * 1024
_MAX_INVENTORY_TEXT_BYTES = 4 * 1024 * 1024
_MAX_PAGES = 3
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_BLOCK_TAGS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "li", "dt", "dd", "tr", "td", "th", "br"}
_IGNORED_TAGS = {"script", "style", "nav", "aside", "header", "footer", "button", "svg"}


class _HTMLRedirect(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.target: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        attributes = dict(attrs)
        if (attributes.get("http-equiv") or "").casefold() == "refresh":
            refresh = re.fullmatch(r"\s*0(?:\.0+)?\s*;\s*url\s*=\s*(.+?)\s*",
                                   attributes.get("content") or "", re.IGNORECASE)
            if refresh:
                self.target = refresh[1].strip("\"'")


class _OfficialHTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool, bool]] = []
        self.title: list[str] = []
        self.parts: list[tuple[str, bool]] = []
        self.definitions: list[tuple[str, int, int]] = []
        self.definition_start: tuple[str, int] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        ignored = tag in _IGNORED_TAGS or any(frame[2] for frame in self.stack)
        main = not ignored and (tag in {"main", "article"} or attributes.get("role") == "main"
                                or any(frame[1] for frame in self.stack))
        # Keep HTML example boundaries so a ``def`` inside an example does not
        # become the owner of the API prose following the block.
        if main and tag == "pre":
            self.parts.append(("\n```\n", False))
        elif main and tag in _BLOCK_TAGS:
            self.parts.append(("\n", False))
        # CLI options and glossary terms also use dt IDs. Only API identifiers
        # establish ownership; other anchors retain the surrounding document.
        if main and tag == "dt" and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", attributes.get("id") or "",
        ):
            self.definition_start = (attributes["id"], len(self.parts))
        # Sphinx renders the classifier with CSS spacing; text extraction must
        # preserve that separator between a parameter name and its type.
        if main and "classifier" in (attributes.get("class") or "").split():
            self.parts.append((": ", False))
        if tag not in _VOID_TAGS:
            self.stack.append((tag, main, ignored))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        position = next((index for index in range(len(self.stack) - 1, -1, -1)
                         if self.stack[index][0] == tag), None)
        if position is None:
            return
        if tag == "dt" and self.definition_start is not None:
            name, start = self.definition_start
            self.definitions.append((name, start, len(self.parts)))
            self.definition_start = None
        if self.stack[position][1] and tag == "pre":
            self.parts.append(("\n```\n", False))
        elif self.stack[position][1] and tag in _BLOCK_TAGS:
            self.parts.append(("\n", False))
        del self.stack[position:]

    def handle_data(self, data: str) -> None:
        if any(frame[0] == "title" for frame in self.stack):
            self.title.append(data)
        if not self.stack or not self.stack[-1][1] or self.stack[-1][2]:
            return
        preserve = any(frame[0] == "pre" for frame in self.stack)
        self.parts.append((data if preserve else re.sub(r"\s+", " ", data), preserve))

    def text(self) -> str:
        return self._render_parts(self.parts)

    def definition_owners(self, text: str) -> dict[str, str]:
        """Bind source IDs to unchanged declaration text in document order."""
        owners = {}
        cursor = 0
        for name, start, end in self.definitions:
            declaration = self._render_parts(self.parts[start:end]).strip()
            if not declaration:
                continue
            position = text.find(declaration, cursor)
            if position >= 0:
                owners[str(position)] = name
                cursor = position + len(declaration)
        return owners

    @staticmethod
    def _render_parts(parts: list[tuple[str, bool]]) -> str:
        rendered = []
        for preserve, group in groupby(parts, key=lambda part: part[1]):
            text = "".join(part[0] for part in group)
            if not preserve:
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r" *\n *", "\n", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
            rendered.append(text)
        return "".join(rendered).strip("\n")


def _read_official_resource(
    url: str, *, timeout_seconds: int, allowed_domains: set[str], accept: str,
) -> tuple[bytes, str, str]:
    """Read one bounded resource, validating each requested URL at the HTTP boundary."""
    for _ in range(3):
        if not is_allowed_doc_url(url) or not result_matches_domains(url, allowed_domains):
            raise ValueError("official resource is outside documentation scope")
        timeout = bounded_request_timeout(timeout_seconds, stage="retrieval")
        with requests.get(url, timeout=timeout, stream=True, allow_redirects=False,
                          headers={"Accept": accept, "User-Agent": "DocuMate/0.2"}) as response:
            check_request_deadline("retrieval")
            if response.is_redirect:
                target = urljoin(url, response.headers.get("Location", ""))
                if target == url:
                    raise ValueError("official resource redirected to itself")
                url = target
                continue
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("official resource did not return a complete response")
            if accept == "text/html" and "html" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("official page did not return HTML")
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_content(chunk_size=65536):
                check_request_deadline("retrieval")
                size += len(chunk)
                if size > _MAX_PAGE_BYTES:
                    raise ValueError("official page exceeds capture limit")
                chunks.append(chunk)
            check_request_deadline("retrieval")
            raw = b"".join(chunks)
            encoding = response.encoding if "charset=" in response.headers.get("Content-Type", "").lower() else "utf-8"
            if accept == "text/html":
                # Some stable aliases (including PyTorch) are HTML redirect
                # pages. Read their declarative target without executing script.
                redirect = _HTMLRedirect()
                redirect.feed(raw.decode(encoding or "utf-8", errors="replace"))
                redirect.close()
                if redirect.target is not None:
                    target = urljoin(url, redirect.target)
                    if target == url:
                        raise ValueError("official resource redirected to itself")
                    url = target
                    continue
            return raw, url, encoding or "utf-8"
    raise ValueError("official resource exceeded redirect limit")


def _capture_page(original_url: str, *, timeout_seconds: int, allowed_domains: set[str]) -> EvidenceRef:
    # A full-page capture must not inherit a search result's unrelated anchor.
    source_url = urldefrag(original_url)[0]
    raw, url, encoding = _read_official_resource(
        source_url, timeout_seconds=timeout_seconds, allowed_domains=allowed_domains, accept="text/html",
    )
    html = raw.decode(encoding, errors="replace")
    parser = _OfficialHTMLText()
    parser.feed(html)
    parser.close()
    text = parser.text()
    if not text.strip():
        raise ValueError("official page has no recognized main content")
    snapshot = build_snapshot(
        source_uri=source_url, title="".join(parser.title).strip() or source_url,
        media_type="text/html", source_type="official", content=raw, parser="official_html", parser_version="3",
        parser_config={"content_region": "main", "resolved_url": url}, capture_scope="full_document",
        quality_issues=["html_structure_flattened"],
    )
    element = DocumentElement(element_id=f"{snapshot.snapshot_id}:body", kind="paragraph", text=text,
                              metadata={"definition_owners": parser.definition_owners(text)},
                              anchors=[SourceAnchor(kind="web", start=0, end=len(text), precision="element")])
    return build_evidence(snapshot=snapshot, element=element)


def _inventory_pages(
    requirement: RetrievalRequirement, source_candidates: list[SearchHit], *,
    timeout_seconds: int, allowed_domains: set[str],
) -> list[str]:
    """Resolve Python objects from the versioned Sphinx inventory published by their project."""
    library = (requirement.library or "").casefold()
    if not requirement.symbols or (requirement.version is not None
                                  and not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", requirement.version)):
        return []
    if library == "python":
        locale = ""
        for hit in source_candidates:
            parsed = urlparse(hit.evidence.snapshot.source_uri)
            match = re.match(r"/([a-z]{2,3}(?:-[a-z]{2})?/)?[0-9]+(?:\.[0-9]+)*/", parsed.path)
            if parsed.hostname == "docs.python.org" and match:
                locale = match.group(1) or ""
                break
        root = f"https://docs.python.org/{locale}{requirement.version or '3'}/"
    elif library == "pytorch":
        root = f"https://docs.pytorch.org/docs/{requirement.version or 'stable'}/"
    else:
        return []
    raw, inventory_url, _ = _read_official_resource(
        root + "objects.inv", timeout_seconds=timeout_seconds, allowed_domains=allowed_domains,
        accept="application/octet-stream",
    )
    header = raw.split(b"\n", 4)
    if len(header) != 5 or header[0].rstrip(b"\r") != b"# Sphinx inventory version 2":
        raise ValueError("unsupported official inventory format")
    decompressor = zlib.decompressobj()
    text = decompressor.decompress(header[4], _MAX_INVENTORY_TEXT_BYTES + 1)
    if len(text) > _MAX_INVENTORY_TEXT_BYTES or not decompressor.eof:
        raise ValueError("official inventory is incomplete or exceeds capture limit")
    entries: dict[str, str] = {}
    for line in text.decode("utf-8").splitlines():
        entry = re.fullmatch(r"(.+?)\s+(\S+)\s+(-?\d+)\s+(\S+)\s+(.*)", line)
        if entry and entry[2].startswith("py:"):
            location = entry[4][:-1] + entry[1] if entry[4].endswith("$") else entry[4]
            entries[entry[1]] = urljoin(inventory_url, location)
    pages = []
    for symbol in requirement.symbols:
        name = symbol.removeprefix("builtins.") if library == "python" else symbol
        url = entries.get(name)
        if url is None:
            # Unqualified class names are usable only when the official index
            # identifies one object, never by guessing a module or API filename.
            matches = [value for key, value in entries.items() if key.endswith("." + name)]
            url = matches[0] if len(matches) == 1 else None
        if url is not None and is_allowed_doc_url(url) and result_matches_domains(url, allowed_domains):
            pages.append(urldefrag(url)[0])
    return list(dict.fromkeys(pages))


def _python_module_url(source_url: str, symbols: list[str], version: str) -> str | None:
    """Resolve a discovered stdlib module at the requested documentation version."""
    parsed = urlparse(source_url)
    if parsed.hostname != "docs.python.org":
        return None
    path = re.fullmatch(
        r"/(?P<locale>[a-z]{2,3}(?:-[a-z]{2})?/)?[0-9]+(?:\.[0-9]+)*/library/"
        r"(?P<module>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\.html",
        parsed.path,
    )
    if path is None:
        return None
    module = path.group("module")
    if not any(symbol == module or symbol.startswith(module + ".") for symbol in symbols):
        return None
    return parsed._replace(path=f"/{path.group('locale') or ''}{version}/library/{module}.html",
                           query="", fragment="").geturl()


def refresh_official_sources(
    hits: list[SearchHit], *, source_candidates: list[SearchHit], requirement: RetrievalRequirement,
    timeout_seconds: int, allowed_domains: list[str], warnings: list[str],
) -> list[SearchHit]:
    """Capture the requested API and at most three pages in total."""
    pages: dict[str, SearchHit | None] = {}
    domains = normalized_domain_set(allowed_domains)
    library = (requirement.library or "").casefold()
    try:
        pages.update((url, None) for url in _inventory_pages(
            requirement, source_candidates, timeout_seconds=timeout_seconds, allowed_domains=domains,
        ))
    except (requests.RequestException, ValueError, zlib.error):
        check_request_deadline("retrieval")
        warnings.append("official_inventory_unavailable")
    if requirement.version is None and library in {"pandas", "numpy"}:
        # These projects publish named APIs at stable routes. Fetch a new
        # source even when search returns only archives or adjacent APIs;
        # never relabel an archived excerpt as the current documentation.
        api_root = ("https://pandas.pydata.org/docs/reference/api" if library == "pandas"
                    else "https://numpy.org/doc/stable/reference/generated")
        for symbol in requirement.symbols:
            if not re.fullmatch(rf"{library}(?:\.[A-Za-z_][A-Za-z0-9_]*)+", symbol):
                continue
            url = f"{api_root}/{symbol}.html"
            if is_current_stable_doc_url(url) and result_matches_domains(url, domains):
                pages[url] = None
        if library == "pandas":
            # A discovered guide can supply conditions omitted by the API
            # reference. Read its current publication rather than mixing an
            # archived guide into the current reference's evidence packet.
            for candidate in source_candidates:
                parsed = urlparse(candidate.evidence.snapshot.source_uri)
                page = re.fullmatch(r"/pandas-docs/version/[0-9]+(?:\.[0-9]+)*/(.+\.html)", parsed.path)
                if parsed.hostname != "pandas.pydata.org" or page is None:
                    continue
                url = parsed._replace(path=f"/docs/{page[1]}", fragment="").geturl()
                if is_current_stable_doc_url(url) and result_matches_domains(url, domains):
                    pages.setdefault(url, None)
    elif library == "python" and (requirement.version is None
                                  or re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", requirement.version)):
        # Ranking can spend its budget on unrelated moving aliases before a
        # versioned module gets a slot. Use the discovered module only to find
        # a new source at the requested version, or /3/ when unspecified.
        for candidate in source_candidates:
            url = _python_module_url(candidate.evidence.snapshot.source_uri, requirement.symbols,
                                     requirement.version or "3")
            if url is not None and is_allowed_doc_url(url) and result_matches_domains(url, domains):
                pages.setdefault(url, None)
    for hit in hits:
        url = urldefrag(hit.evidence.snapshot.source_uri)[0]
        if url in pages or (requirement.version is None and is_current_stable_doc_url(url)):
            if pages.get(url) is None:
                pages[url] = hit
    captured: dict[str, SearchHit] = {}
    for url, hit in list(pages.items())[:_MAX_PAGES]:
        try:
            evidence = _capture_page(hit.evidence.snapshot.source_uri if hit else url,
                                     timeout_seconds=timeout_seconds, allowed_domains=domains)
        except (requests.RequestException, ValueError):
            check_request_deadline("retrieval")
            if "official_page_refresh_failed" not in warnings:
                warnings.append("official_page_refresh_failed")
            continue
        captured[url] = (hit.model_copy(update={"evidence": evidence}) if hit else SearchHit(
            evidence=evidence, rank=1, score=RetrievalScore(metric="direct_official_page", direction="higher"),
        ))
    result = []
    emitted_hashes = set()
    for hit in captured.values():
        content_hash = hit.evidence.snapshot.content_hash
        if content_hash not in emitted_hashes:
            result.append(hit)
            emitted_hashes.add(content_hash)
    for hit in hits:
        url = urldefrag(hit.evidence.snapshot.source_uri)[0]
        if url in captured:
            continue
        evidence = hit.evidence
        if (requirement.version is None and url in pages and evidence.snapshot.parser == "tavily"
                and "freshness_unverified" not in evidence.snapshot.quality_issues):
            snapshot = evidence.snapshot.model_copy(update={
                "quality_issues": [*evidence.snapshot.quality_issues, "freshness_unverified"],
            })
            hit = hit.model_copy(update={"evidence": build_evidence(
                snapshot=snapshot, element=evidence.element, start=evidence.selection.start, end=evidence.selection.end,
            )})
        result.append(hit)
    return result
