from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from src.app.web.streamlit_api_client import AgentCallResult, AgentStreamEvent
from src.app.web.streamlit_sources import render_code, render_evidence
from src.app.web.streamlit_state import ChatMessage
from src.core.answer_schema import AnswerResponse, ContentUnit, finalize_answer, iter_content_units, text_document


_DEFAULT_ERROR_MESSAGE = "응답을 받지 못했습니다."
_DEFAULT_PROCESSING_MESSAGE = "응답을 준비하고 있습니다."
_STAGE_MESSAGES = {
    "summarize": "이전 맥락을 정리하고 있습니다.",
    "planner": "질문을 분석하고 있습니다.",
    "retrieval": "근거를 수집하고 있습니다.",
    "pre_synthesis_validation": "검색 결과를 확인하고 있습니다.",
    "synthesis": "답변을 작성하고 있습니다.",
    "post_synthesis_validation": "답변의 근거를 확인하고 있습니다.",
    "validation": "근거를 검증하고 있습니다.",
    "action_postprocess": "결과를 정리하고 있습니다.",
}


def render_chat_history(messages: list[ChatMessage], fastapi_url: str) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer_response(message["response"], fastapi_url)


def render_answer_response(response: AnswerResponse, fastapi_url: str) -> None:
    response = AnswerResponse.model_validate(response.model_dump(mode="json"))
    citations = {citation.evidence.id: citation for citation in response.citations}
    checks = {check.unit_id: check for check in response.checks}
    for issue in response.issues:
        if issue.unit_id is None:
            st.warning(issue.message)

    for index, block in enumerate(response.content.blocks):
        prefix = f"b{index}."
        units = [(path, unit) for path, unit in iter_content_units(response.content) if path.startswith(prefix)]
        if block.type == "paragraph":
            st.markdown(" ".join(_unit_text(unit, citations) for unit in block.content))
        elif block.type == "heading":
            st.markdown("#" * block.level + " " + _unit_text(block.content, citations))
        elif block.type == "list":
            st.markdown("\n".join(
                f"{item_index + 1}. {_unit_text(unit, citations)}" if block.ordered
                else f"- {_unit_text(unit, citations)}"
                for item_index, unit in enumerate(block.items)
            ))
        elif block.type == "code":
            render_code(block.content.text, language=block.language or "text")
            _render_basis(block.content)
        elif block.type == "table":
            header = "| " + " | ".join(_table_cell(unit, citations) for unit in block.columns) + " |"
            separator = "| " + " | ".join("---" for _ in block.columns) + " |"
            rows = ["| " + " | ".join(_table_cell(unit, citations) for unit in row) + " |" for row in block.rows]
            st.markdown("\n".join([header, separator, *rows]))

        paths = {path for path, _ in units}
        for issue in response.issues:
            if issue.unit_id in paths:
                st.warning(issue.message)
        for path, unit in units:
            check = checks.get(path)
            if check is not None and check.reference_status == "missing":
                st.warning("이 내용에 연결된 근거를 찾지 못했습니다.")
            elif check is not None and check.support_status == "unsupported":
                st.warning("이 인용문은 보관한 원문과 일치하지 않습니다.")
        _render_citations([unit for _, unit in units], citations)

    if response.citations:
        with st.expander("근거 확인 범위"):
            resolved = sum(check.reference_status == "resolved" for check in response.checks)
            exact = sum(check.support_status == "exact_match" for check in response.checks)
            st.caption(f"원문에 연결된 내용 {resolved}개 · 원문과 일치하는 발췌 {exact}개")
            if any(check.reference_status == "resolved" and check.support_status == "not_evaluated" for check in response.checks):
                st.caption("출처 연결은 확인했지만, 출처가 모든 설명이나 해석을 뒷받침하는지는 별도로 평가하지 않았습니다.")

    for action in response.actions:
        label = "파일 저장" if action.kind == "save_text" else "Slack 전송"
        if action.status == "error":
            st.error(f"{label} 실패: {action.error or action.message or '처리하지 못했습니다.'}")
        elif action.status == "skipped":
            st.info(f"{label} 보류: {action.message or '실행 조건을 확인해 주세요.'}")
        else:
            detail = action.message or action.target or ""
            st.success(f"{label} 완료" + (f": {detail}" if detail else ""))
            if action.kind == "save_text" and action.file_path:
                filename = Path(action.file_path).name
                url = f"{fastapi_url}/download/{quote(filename, safe='')}"
                st.markdown(f"[파일 다운로드 ({filename})]({url})")


def _unit_text(unit: ContentUnit, citations: dict) -> str:
    prefix = {"inference": "**해석** · ", "example": "**예시** · ", "excerpt": "**원문 발췌** · "}.get(unit.basis, "")
    labels = " ".join(f"[{citations[ref].number}]" for ref in unit.refs if ref in citations)
    return prefix + unit.text + (f" {labels}" if labels else "")


def _table_cell(unit: ContentUnit, citations: dict) -> str:
    return _unit_text(unit, citations).replace("|", "\\|").replace("\n", "<br>")


def _render_basis(unit: ContentUnit) -> None:
    if unit.basis == "excerpt":
        st.caption("원문 발췌")
        return
    label = {"inference": "해석에 따라 작성한 코드", "example": "코드 예시", "source": "근거를 바탕으로 작성한 코드"}.get(unit.basis, "코드")
    st.caption(label + " · 실행 확인 안 됨")


def _render_citations(units: list[ContentUnit], citations: dict) -> None:
    refs = list(dict.fromkeys(ref for unit in units for ref in unit.refs if ref in citations))
    if not refs:
        return
    with st.container(horizontal=True):
        for ref in refs:
            citation = citations[ref]
            with st.popover(f"근거 [{citation.number}]", help=citation.evidence.snapshot.title):
                render_evidence(citation.evidence)


def process_chat_prompt(
    prompt: str,
    append_user_message: Callable[[ChatMessage], None],
    append_assistant_message: Callable[[ChatMessage], None],
    stream_agent: Callable[[str], Iterable[AgentStreamEvent]],
) -> None:
    append_user_message({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result: AgentCallResult | None = None
    last_error_message = _DEFAULT_ERROR_MESSAGE
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("요청을 접수했습니다.")
        for event in stream_agent(prompt):
            if event.event == "final_response" and event.result is not None:
                result = event.result
            elif event.event == "error":
                last_error_message = str(event.data.get("message") or "응답 처리 중 오류가 발생했습니다.")
                status_placeholder.markdown(last_error_message)
            else:
                progress = _progress_message_for_event(event)
                if progress:
                    status_placeholder.markdown(progress)
        status_placeholder.empty()

    if result is None:
        result = AgentCallResult(response=finalize_answer(text_document(last_error_message), []))
    append_assistant_message({"role": "assistant", "response": result.response})
    st.rerun()


def _progress_message_for_event(event: AgentStreamEvent) -> str:
    if event.event == "request_started":
        return "요청을 접수했습니다."
    if event.event == "progress_snapshot":
        return str(event.data.get("summary") or "").strip() or _DEFAULT_PROCESSING_MESSAGE
    if event.event not in {"stage_started", "heartbeat"}:
        return ""
    return _STAGE_MESSAGES.get(str(event.data.get("stage") or "").strip(), _DEFAULT_PROCESSING_MESSAGE)
