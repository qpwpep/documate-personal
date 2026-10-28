"""Refresh bounded stable documentation candidates from their current pages."""

from __future__ import annotations

from html.parser import HTMLParser
from itertools import groupby
import re
from urllib.parse import urldefrag, urljoin

import requests

from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import EvidenceRef, RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import RetrievalRequirement
from src.infra.tools.docs_search.policy import (
    is_allowed_doc_url, is_current_stable_doc_url, normalized_domain_set, result_matches_domains,
)


_MAX_PAGE_BYTES = 2 * 1024 * 1024
_MAX_PAGES = 3
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_BLOCK_TAGS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "li", "dt", "dd", "tr", "td", "th", "br"}
_IGNORED_TAGS = {"script", "style", "nav", "aside", "header", "footer", "button", "svg"}


class _OfficialHTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool, bool]] = []
        self.title: list[str] = []
        self.parts: list[tuple[str, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        ignored = tag in _IGNORED_TAGS or any(frame[2] for frame in self.stack)
        main = not ignored and (tag in {"main", "article"} or attributes.get("role") == "main"
                                or any(frame[1] for frame in self.stack))
        if main and tag in _BLOCK_TAGS:
            self.parts.append(("\n", False))
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
        if self.stack[position][1] and tag in _BLOCK_TAGS:
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
        rendered = []
        for preserve, parts in groupby(self.parts, key=lambda part: part[1]):
            text = "".join(part[0] for part in parts)
            if not preserve:
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r" *\n *", "\n", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
            rendered.append(text)
        return "".join(rendered).strip("\n")


def _capture_page(original_url: str, *, timeout_seconds: int, allowed_domains: set[str]) -> EvidenceRef:
    url = urldefrag(original_url)[0]
    for _ in range(3):
        with requests.get(url, timeout=timeout_seconds, stream=True, allow_redirects=False,
                          headers={"Accept": "text/html", "User-Agent": "DocuMate/0.2"}) as response:
            if response.is_redirect:
                target = urljoin(url, response.headers.get("Location", ""))
                if (target == url or not is_allowed_doc_url(target)
                        or not result_matches_domains(target, allowed_domains)):
                    raise ValueError("official page redirected outside documentation scope")
                url = target
                continue
            response.raise_for_status()
            if response.status_code != 200 or "html" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("official page did not return HTML")
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > _MAX_PAGE_BYTES:
                    raise ValueError("official page exceeds capture limit")
                chunks.append(chunk)
            raw = b"".join(chunks)
            encoding = response.encoding if "charset=" in response.headers.get("Content-Type", "").lower() else "utf-8"
            html = raw.decode(encoding or "utf-8", errors="replace")
            break
    else:
        raise ValueError("official page exceeded redirect limit")
    parser = _OfficialHTMLText()
    parser.feed(html)
    parser.close()
    text = parser.text()
    if not text.strip():
        raise ValueError("official page has no recognized main content")
    snapshot = build_snapshot(
        source_uri=original_url, title="".join(parser.title).strip() or original_url,
        media_type="text/html", source_type="official", content=raw, parser="official_html", parser_version="1",
        parser_config={"content_region": "main", "resolved_url": url}, capture_scope="full_document",
        quality_issues=["html_structure_flattened"],
    )
    element = DocumentElement(element_id=f"{snapshot.snapshot_id}:body", kind="paragraph", text=text,
                              anchors=[SourceAnchor(kind="web", start=0, end=len(text), precision="element")])
    return build_evidence(snapshot=snapshot, element=element)


def refresh_stable_sources(
    hits: list[SearchHit], *, requirement: RetrievalRequirement,
    timeout_seconds: int, allowed_domains: list[str], warnings: list[str],
) -> list[SearchHit]:
    """Capture the requested stable API and at most three pages in total."""
    stable: dict[str, SearchHit | None] = {}
    domains = normalized_domain_set(allowed_domains)
    if requirement.version is None and str(requirement.library or "").casefold() == "pandas":
        # pandas publishes each named API at this stable route. Fetch a new
        # source even when search returns only archives or adjacent APIs;
        # never relabel an archived excerpt as the current documentation.
        for symbol in requirement.symbols:
            if not re.fullmatch(r"pandas(?:\.[A-Za-z_][A-Za-z0-9_]*)+", symbol):
                continue
            url = f"https://pandas.pydata.org/docs/reference/api/{symbol}.html"
            if is_current_stable_doc_url(url) and result_matches_domains(url, domains):
                stable[url] = None
    for hit in hits:
        if is_current_stable_doc_url(hit.evidence.snapshot.source_uri):
            url = urldefrag(hit.evidence.snapshot.source_uri)[0]
            if stable.get(url) is None:
                stable[url] = hit
    captured: dict[str, SearchHit] = {}
    for url, hit in list(stable.items())[:_MAX_PAGES]:
        try:
            evidence = _capture_page(hit.evidence.snapshot.source_uri if hit else url,
                                     timeout_seconds=timeout_seconds, allowed_domains=domains)
        except (requests.RequestException, ValueError):
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
        if url in stable and evidence.snapshot.parser == "tavily" and "freshness_unverified" not in evidence.snapshot.quality_issues:
            snapshot = evidence.snapshot.model_copy(update={
                "quality_issues": [*evidence.snapshot.quality_issues, "freshness_unverified"],
            })
            hit = hit.model_copy(update={"evidence": build_evidence(
                snapshot=snapshot, element=evidence.element, start=evidence.selection.start, end=evidence.selection.end,
            )})
        result.append(hit)
    return result
