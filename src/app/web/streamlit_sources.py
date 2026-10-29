from __future__ import annotations

import re
from dataclasses import replace
from html import escape
from textwrap import dedent
from urllib.parse import quote, urljoin, urlparse

import streamlit as st
from markdown_it import MarkdownIt
from markdown_it.token import Token

from src.core.answer_schema import CodeBlock
from src.core.documents import SourceAnchor, TableData
from src.core.evidence import EvidenceRef, selected_source_anchors, source_line_records
from src.core.python_source import PythonSourceMap


def render_evidence(
    evidence: EvidenceRef, *, cited_texts: list[str] | None = None,
    answer_context: list[str | CodeBlock] | None = None,
) -> None:
    """Show the retained source, independently of the current uploaded file."""
    snapshot = evidence.snapshot
    element = evidence.element
    st.markdown(
        '<div class="dm-source-header">'
        f'<p class="dm-source-title" tabindex="-1"><strong>{escape(snapshot.title or snapshot.source_uri)}</strong></p>'
        '<button type="button" class="dm-source-close" aria-label="근거 닫기">닫기</button></div>',
        unsafe_allow_html=True,
    )
    source_uri = snapshot.source_uri
    parsed = urlparse(source_uri)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        safe_url = quote(source_uri, safe=":/?#[]@!$&'*,;=+-._~%")
        st.markdown(f"[현재 출처 페이지 열기]({safe_url})")
    _render_answer_context(answer_context if answer_context is not None else cited_texts or [])
    lines = source_line_records(evidence)
    st.markdown("**답변에 사용한 부분**")
    code_ranges = _matching_code_ranges(evidence, cited_texts or []) if element.kind == "code" else []
    if element.kind == "code":
        if lines:
            matched = [index for index, line in enumerate(lines)
                       if any(start < line["end"] and line["start"] < end for start, end in code_ranges)]
            visible = {neighbor for index in matched for neighbor in range(max(0, index - 2), min(len(lines), index + 3))}
            if matched:
                st.caption("답변의 코드와 일치하는 원문: " + ", ".join(f"{lines[index]['line_number']}행" for index in matched))
                if len(code_ranges) > 1:
                    st.caption("일치하는 위치를 모두 표시했습니다.")
            st.html(_code_lines_html([line for index, line in enumerate(lines) if not visible or index in visible], code_ranges))
        else:
            # Unknown physical provenance stays unnumbered.
            render_code(evidence.excerpt, language=element.language or "text")
    elif snapshot.source_type == "official" and element.table is None:
        _render_official_excerpt(evidence, cited_texts or [])
    else:
        st.text(evidence.excerpt)
    if evidence.selection.cell_ids:
        st.caption("선택한 표 셀: " + ", ".join(evidence.selection.cell_ids))
    else:
        start = evidence.selection.start
        end = evidence.selection.end if evidence.selection.end is not None else len(element.text)
        st.caption(_selection_location(evidence, start=start, end=end))
    if element.heading_path:
        st.caption("문서 위치: " + " › ".join(element.heading_path))

    source_status = ["공식 문서" if snapshot.source_type == "official" else "업로드 자료"]
    if snapshot.capture_scope == "provider_excerpt":
        source_status.append("일부 수집")
    if "freshness_unverified" in snapshot.quality_issues:
        source_status.append("최신성 미확인")
    st.caption(" · ".join(source_status))
    with st.expander("출처와 수집 정보"):
        st.caption(f"원본 자료: {source_uri}")
        if element.table is not None and evidence.selection.cell_ids:
            selected_ids = set(evidence.selection.cell_ids)
            if any(not cell.anchors for cell in element.table.cells if cell.cell_id in selected_ids) and element.anchors:
                st.caption("일부 선택 셀의 위치가 없어 표 전체 위치를 함께 표시합니다.")
        mapping_anchor = _mapped_line_anchor(evidence, lines)
        for anchor in selected_source_anchors(evidence):
            location = _format_anchor(anchor)
            if anchor.line_start is not None and anchor != mapping_anchor:
                location = "원문 요소 위치: " + location
            st.caption(location)
        if snapshot.capture_scope == "provider_excerpt":
            st.caption("원문의 일부만 수집한 자료입니다. 답변에는 당시 수집한 내용을 사용했습니다.")
        quality_messages = {
            "html_structure_flattened": "원문 웹페이지에서 본문을 추출했습니다. 표와 배치 등 일부 서식은 원본 페이지와 다를 수 있습니다.",
            "freshness_unverified": "현재 공식 페이지를 확인하지 못해 검색에 보관된 내용을 사용했습니다. 최신 문서와 다를 수 있습니다.",
        }
        for issue in snapshot.quality_issues:
            st.warning(quality_messages.get(issue, issue))

    full_source_label = "당시 원문과 위치 보기"
    if code_ranges:
        full_source_label = "전체 원문 보기"
    elif snapshot.source_type == "official":
        full_source_label = "수집한 원문 그대로 보기"
    with st.expander(full_source_label):
        if element.table is not None:
            st.html(_table_html(element.table, selected=set(evidence.selection.cell_ids)))
        elif element.kind == "code":
            st.caption("원문 복사에는 아래 코드 블록의 복사 버튼을 사용하세요. 줄 번호와 강조 표시는 포함되지 않습니다.")
            render_code(element.text, language=element.language or "text")
        elif snapshot.source_type == "official":
            render_code(element.text, language="markdown")
        else:
            st.text(element.text)
        st.caption("이 내용은 답변을 만들 때 보관한 원문입니다. 현재 파일의 변경이나 삭제와 무관합니다.")
        st.caption(f"자료 내용 해시: {snapshot.content_hash}")


def _render_answer_context(items: list[str | CodeBlock]) -> None:
    parser = MarkdownIt("commonmark", {"html": False, "breaks": True})
    rendered = []
    for item in items:
        if isinstance(item, CodeBlock):
            # Preserve code punctuation and whitespace; Markdown can turn a
            # Python comment into a heading or strip an identifier's underscores.
            code = f'<code style="white-space:pre-wrap;overflow-wrap:anywhere">{escape(item.content.text)}</code>'
            content = f"<p>{code}</p>" if item.inline else f"<pre>{code}</pre>"
        elif item.strip():
            content = parser.render(item)
        else:
            continue
        rendered.append(f'<div class="dm-source-answer-text">{content}</div>')
    rendered = list(dict.fromkeys(rendered))
    if not rendered:
        return
    # This is already escaped HTML. Markdown's highlighter would replace the
    # code wrapping styles and force horizontal scrolling on narrow screens.
    st.html(
        '<div class="dm-source-answer-context"><p class="dm-source-context-label">이 근거가 연결된 답변</p>'
        + "".join(rendered[:2]) + '</div>',
    )
    if len(rendered) > 2:
        with st.expander(f"함께 연결된 답변 더 보기 ({len(rendered) - 2}개)"):
            for content in rendered[2:]:
                st.html(content)


def _render_official_excerpt(evidence: EvidenceRef, cited_texts: list[str]) -> None:
    """Shorten the initial view without narrowing the retained evidence range."""
    excerpt = evidence.excerpt
    provider_excerpt = evidence.snapshot.capture_scope == "provider_excerpt"
    if len(excerpt) <= 700 and len(excerpt.splitlines()) <= 12:
        _render_source_markdown(excerpt, source_uri=evidence.snapshot.source_uri, provider_excerpt=provider_excerpt)
        return

    # Only whole answer phrases qualify; shared identifiers or inferred meaning
    # do not establish which part of a documentation range supports an answer.
    exact_phrases = list(dict.fromkeys(text for text in cited_texts if text.strip() and text in excerpt))
    if exact_phrases:
        st.caption("답변 문구와 정확히 일치하는 원문을 먼저 표시합니다. 연결된 전체 범위는 아래에서 확인하세요.")
        for phrase in exact_phrases:
            st.html('<div class="dm-source-markdown" style="white-space:pre-wrap;overflow-wrap:anywhere">'
                    '<mark style="background:#ffec99;color:#1a1a1a">' + escape(phrase) + '</mark></div>')
    else:
        st.caption("이 설명에는 넓은 원문 범위가 연결돼 있습니다. 특정 문장과의 일치 위치를 확인한 것은 아닙니다.")
        st.caption("원문 앞부분 미리보기 · 연결된 전체 발췌는 아래에서 확인하세요.")
        preview = "\n".join(excerpt.splitlines()[:6])[:420]
        st.text(preview + "…")
    with st.expander("답변에 사용한 전체 발췌"):
        _render_source_markdown(excerpt, source_uri=evidence.snapshot.source_uri, provider_excerpt=provider_excerpt)


def _render_source_markdown(text: str, *, source_uri: str, provider_excerpt: bool) -> None:
    """Resolve parsed links against their source without rewriting captured text."""
    parser = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])
    tokens = parser.parse(_source_reader_markdown(text, parser))
    if provider_excerpt:
        _separate_flattened_headings(tokens)
    pending = list(tokens)
    url_attributes = {"link_open": "href", "image": "src"}
    while pending:
        token = pending.pop()
        pending.extend(token.children or [])
        attribute = url_attributes.get(token.type)
        if attribute is not None:
            target = token.attrGet(attribute)
            if target is not None:
                absolute = urljoin(source_uri, target)
                if parser.validateLink(absolute):
                    token.attrSet(attribute, absolute)
    content = parser.renderer.render(tokens, parser.options, {})
    st.html(
        '<style>.dm-source-markdown {overflow-wrap:anywhere;max-width:100%}'
        '.dm-source-markdown :is(h1,h2,h3,h4,h5,h6) {line-height:1.4;margin:0.75rem 0 0.4rem}'
        '.dm-source-markdown h1 {font-size:1.25rem}'
        '.dm-source-markdown h2 {font-size:1.125rem}'
        '.dm-source-markdown :is(h3,h4,h5,h6) {font-size:1rem}'
        '.dm-source-markdown pre {overflow-x:auto;max-width:100%;padding:0.75rem;'
        'border:1px solid currentColor;border-radius:0.4rem;white-space:pre}'
        '.dm-source-markdown code {font-family:monospace}'
        '.dm-source-markdown pre code {white-space:pre;overflow-wrap:normal}'
        '.dm-source-markdown img {max-width:100%;height:auto}</style>'
        '<div class="dm-source-markdown">' + content + '</div>'
    )


def _separate_flattened_headings(tokens: list[Token]) -> None:
    """Recover a provider-flattened Sphinx heading at its permalink boundary."""
    for index in reversed(range(len(tokens) - 2)):
        opening, inline, closing = tokens[index:index + 3]
        if opening.type != "heading_open" or opening.level != 0 or inline.children is None:
            continue
        nesting = 0
        for child_index, child in enumerate(inline.children):
            nesting += child.nesting
            # A literal inside code, emphasis or a link is not a block boundary.
            if nesting or child.type != "text":
                continue
            title, separator, body = child.content.partition("#. ")
            if not separator:
                continue
            title_children = inline.children[:child_index] + [replace(child, content=title)]
            body_children = [replace(child, content=body)] + inline.children[child_index + 1:]
            title_text = "".join(item.content for item in title_children)
            body_text = "".join(item.content for item in body_children)
            if not title_text.strip() or not body_text.strip():
                continue
            tokens[index:index + 3] = [
                opening, replace(inline, content=title_text, children=title_children), closing,
                Token("paragraph_open", "p", 1, block=True),
                replace(inline, content=body_text, children=body_children),
                Token("paragraph_close", "p", -1, block=True),
            ]
            break


def _source_reader_markdown(text: str, parser: MarkdownIt) -> str:
    """Display top-level provider definitions as prose without changing retained source."""
    lines = text.splitlines(keepends=True)
    # Sphinx-derived parameter definitions are not CommonMark. A leading colon
    # and indentation otherwise turn the next parameter into a code block.
    definition = re.compile(r"(?m)^(?:[ \t]*:[ \t]+)?\*\*[^*\n]+\*\*[^\n]*\n[ \t]*:[ \t]+")
    for token in reversed(parser.parse(text)):
        # Nested token content omits list/quote prefixes, so replacing those
        # source lines would discard their surrounding Markdown structure.
        if (token.type, token.level) not in {("inline", 1), ("code_block", 0)} or token.map is None:
            continue
        # One indented block can contain both definitions and ordinary code.
        # Replace only matching paragraphs, retaining every other source line.
        paragraphs = re.finditer(r"(?m)(?:^[ \t]*\S[^\n]*(?:\n|$))+", token.content)
        for paragraph in reversed(list(paragraphs)):
            # Provider definitions can retain another indentation level after
            # CommonMark removes the first four spaces of a code block.
            paragraph_text = dedent(paragraph.group())
            if definition.search(paragraph_text) is None:
                continue
            display = re.sub(r"(?m)^[ \t]*:[ \t]+", "\n", paragraph_text)
            display = re.sub(r"(?m)^(\*\*[^*\n]+\*\*)(?=[^\s:])", r"\1 ", display)
            start = token.map[0] + token.content.count("\n", 0, paragraph.start())
            end = start + len(paragraph.group().splitlines())
            lines[start:end] = [display.rstrip("\n") + "\n"]
    return "".join(lines)


def _matching_code_ranges(evidence: EvidenceRef, cited_texts: list[str]) -> list[tuple[int, int]]:
    """Locate exact cited code, never infer locations from numbers in prose."""
    snippets = set()
    parser = MarkdownIt()
    for text in cited_texts:
        if text in evidence.excerpt:
            snippets.add(text)
        for token in parser.parse(text):
            snippets.update(child.content for child in token.children or [] if child.type == "code_inline")
    ranges = set()
    end = evidence.selection.end if evidence.selection.end is not None else len(evidence.element.text)
    for snippet in snippets:
        if not snippet:
            continue
        start = evidence.element.text.find(snippet, evidence.selection.start, end)
        while start >= 0:
            ranges.add((start, start + len(snippet)))
            start = evidence.element.text.find(snippet, start + 1, end)
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _code_lines_html(lines: list[dict], ranges: list[tuple[int, int]]) -> str:
    rendered = ['<div class="dm-source-code" style="font-family:monospace;font-size:0.875rem;line-height:1.6;max-width:100%">',
                '<style>.dm-source-code mark {background:#ffec99;color:#1a1a1a}</style>']
    previous = None
    for line in lines:
        number = line["line_number"]
        if previous is not None and number > previous + 1:
            rendered.append('<div aria-label="생략한 원문">…</div>')
        previous = number
        text = line["text"].rstrip("\r\n")
        cursor = 0
        content = []
        for start, end in ranges:
            first, last = max(0, start - line["start"]), min(len(text), end - line["start"])
            if first < last:
                content.extend((escape(text[cursor:first]), "<mark>" + escape(text[first:last]) + "</mark>"))
                cursor = last
        content.append(escape(text[cursor:]))
        rendered.append(
            f'<div role="group" aria-label="원문 {number}행" style="display:grid;grid-template-columns:3.5em minmax(0,1fr);gap:0.7em">'
            f'<span aria-hidden="true" style="user-select:none;text-align:right;opacity:0.7">{number}</span>'
            '<code style="white-space:pre-wrap;overflow-wrap:anywhere;background:transparent;color:inherit;padding:0">'
            + ("".join(content) or "&#8203;") + "</code></div>"
        )
    return "".join(rendered) + "</div>"


def _format_anchor(anchor: SourceAnchor) -> str:
    parts: list[str] = []
    if anchor.cell_index is not None:
        parts.append(f"셀 {anchor.cell_index + 1}")
    if anchor.cell_id:
        parts.append(f"셀 ID {anchor.cell_id}")
    if anchor.line_start is not None:
        parts.append(_line_range(anchor.line_start, anchor.line_end or anchor.line_start))
    if anchor.page_no is not None:
        parts.append(f"{anchor.page_no}페이지")
        if anchor.bbox is None:
            parts.append("페이지 안의 정확한 영역은 제공되지 않았습니다")
        else:
            parts.append(f"영역 {anchor.bbox} ({anchor.coordinate_space}, {anchor.coordinate_origin})")
    if not parts:
        parts.append("문서 요소 위치")
    return " · ".join(parts)


def render_code(text: str, *, language: str, line_numbers: bool = False) -> None:
    # The code widget lays out LF lines. Normalize only its display input;
    # retained evidence and checked answer content keep their original bytes.
    display_text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Streamlit removes one boundary newline; framing preserves blank lines.
    st.code("\n" + display_text + "\n", language=language, line_numbers=line_numbers)


def _line_range(start: int, end: int) -> str:
    return f"{start}행" if start == end else f"{start}–{end}행"


def _mapped_line_anchor(evidence: EvidenceRef, lines: list[dict]) -> SourceAnchor | None:
    if not lines:
        return None
    selected_length = lines[-1]["end"] - lines[0]["start"]
    return next((anchor for anchor in selected_source_anchors(evidence)
                 if anchor.kind in {"code", "notebook", "text"} and anchor.precision == "exact"
                 and anchor.start is not None and anchor.end is not None
                 and anchor.end - anchor.start == selected_length
                 and anchor.line_start == lines[0]["line_number"]
                 and anchor.line_end == lines[-1]["line_number"]), None)


def _selection_location(evidence: EvidenceRef, *, start: int, end: int) -> str:
    lines = source_line_records(evidence)
    if lines:
        line_start, line_end = lines[0]["line_number"], lines[-1]["line_number"]
        anchor = _mapped_line_anchor(evidence, lines)
        scope = f"셀 {anchor.cell_index + 1}의 " if anchor is not None and anchor.cell_index is not None else "원문 "
        return "사용 범위: " + scope + _line_range(line_start, line_end)
    source = PythonSourceMap(evidence.element.text)
    line_start = source.line_at_offset(start)
    line_end = source.line_at_offset(max(start, end - 1))
    return "사용 범위: 수집한 요소의 " + _line_range(line_start, line_end) + " (원문 줄 번호는 확인되지 않았습니다)"


def _table_html(table: TableData, *, selected: set[str]) -> str:
    cells_by_position = {(cell.row, cell.col): cell for cell in table.cells}
    occupied: set[tuple[int, int]] = set()
    rows = max((cell.row + cell.row_span for cell in table.cells), default=0)
    columns = max((cell.col + cell.col_span for cell in table.cells), default=0)
    lines = ['<table style="border-collapse:collapse;width:100%">']
    for row in range(rows):
        lines.append("<tr>")
        for column in range(columns):
            if (row, column) in occupied:
                continue
            cell = cells_by_position.get((row, column))
            if cell is None:
                lines.append("<td></td>")
                continue
            for covered_row in range(row, row + cell.row_span):
                for covered_col in range(column, column + cell.col_span):
                    occupied.add((covered_row, covered_col))
            color = "background:#fff3b0;color:#222;" if cell.cell_id in selected else ""
            lines.append(
                f'<td rowspan="{cell.row_span}" colspan="{cell.col_span}" '
                f'style="border:1px solid #888;padding:0.4rem;white-space:pre-wrap;{color}">'
                f"{escape(cell.text)}</td>"
            )
        lines.append("</tr>")
    lines.append("</table>")
    return "".join(lines)
