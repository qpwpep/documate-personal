from __future__ import annotations

from html import escape
from urllib.parse import quote, urlparse

import streamlit as st

from src.core.documents import SourceAnchor, TableData
from src.core.evidence import EvidenceRef


def render_evidence(evidence: EvidenceRef) -> None:
    """Show the retained source, independently of the current uploaded file."""
    snapshot = evidence.snapshot
    element = evidence.element
    st.markdown(f"**{escape(snapshot.title or snapshot.source_uri)}**")
    source_kind = "공식 문서" if snapshot.source_type == "official" else "업로드 자료"
    st.caption(source_kind)
    if element.heading_path:
        st.caption("문서 위치: " + " › ".join(element.heading_path))
    for anchor in element.anchors:
        st.caption(_format_anchor(anchor))

    if snapshot.capture_scope == "provider_excerpt":
        st.caption("원문의 일부만 수집한 자료입니다. 아래에 당시 수집한 내용을 표시합니다.")
    for issue in snapshot.quality_issues:
        st.warning(str(issue))

    st.markdown("**답변에 사용한 부분**")
    if element.kind == "code":
        render_code(evidence.excerpt, language=element.language or "text", line_numbers=True)
    else:
        st.text(evidence.excerpt)
    if evidence.selection.cell_ids:
        st.caption("선택한 표 셀: " + ", ".join(evidence.selection.cell_ids))
    else:
        start = evidence.selection.start
        end = evidence.selection.end if evidence.selection.end is not None else len(element.text)
        st.caption(_selection_location(evidence, start=start, end=end))

    with st.expander("당시 원문과 위치 보기"):
        if element.table is not None:
            st.html(_table_html(element.table, selected=set(evidence.selection.cell_ids)))
        elif element.kind == "code":
            render_code(element.text, language=element.language or "text", line_numbers=True)
        else:
            st.text(element.text)
        st.caption("이 내용은 답변을 만들 때 보관한 원문입니다. 현재 파일의 변경이나 삭제와 무관합니다.")
        st.caption(f"자료 버전: {snapshot.content_hash}")

    source_uri = snapshot.source_uri
    parsed = urlparse(source_uri)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        safe_url = quote(source_uri, safe=":/?#[]@!$&'*,;=+-._~%")
        st.markdown(f"[현재 원문 열기]({safe_url})")
    else:
        st.caption(f"원본 자료: {source_uri}")


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
    # Streamlit removes one boundary newline; framing preserves the original text.
    st.code("\n" + text + "\n", language=language, line_numbers=line_numbers)


def _line_range(start: int, end: int) -> str:
    return f"{start}행" if start == end else f"{start}–{end}행"


def _selection_location(evidence: EvidenceRef, *, start: int, end: int) -> str:
    text = evidence.element.text
    line_start = text.count("\n", 0, start) + 1
    line_end = text.count("\n", 0, max(start, end - 1)) + 1
    source_start = next(
        (anchor.line_start for anchor in evidence.element.anchors if anchor.line_start is not None),
        None,
    )
    if source_start is not None:
        return "사용 범위: 원문 " + _line_range(source_start + line_start - 1, source_start + line_end - 1)
    return "사용 범위: 수집한 요소의 " + _line_range(line_start, line_end)


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
