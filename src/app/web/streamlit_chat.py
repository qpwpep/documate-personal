from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from html import escape
from urllib.parse import quote

import streamlit as st

from src.app.web.streamlit_api_client import AgentCallResult, AgentStreamEvent
from src.app.web.streamlit_state import ChatMessage


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
            file_path = message.get("file_path", "")
            st.markdown(_clean_saved_file_notice(message["content"], file_path))
            evidence_items = message.get("evidence") or []
            if message["role"] == "assistant" and evidence_items:
                _render_evidence(evidence_items)

            if message["role"] == "assistant" and file_path and os.path.exists(file_path):
                _render_download_button(file_path, fastapi_url)


def process_chat_prompt(
    prompt: str,
    append_user_message: Callable[[ChatMessage], None],
    append_assistant_message: Callable[[ChatMessage], None],
    stream_agent: Callable[[str], Iterable[AgentStreamEvent]] | None = None,
    call_agent: Callable[[str], AgentCallResult] | None = None,
) -> None:
    append_user_message(
        {
            "role": "user",
            "content": prompt,
            "file_path": "",
            "evidence": [],
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    result: AgentCallResult | None = None
    last_error_message = _DEFAULT_ERROR_MESSAGE
    with st.chat_message("assistant"):
        status_placeholder = st.empty() if hasattr(st, "empty") else st
        status_placeholder.markdown("요청을 접수했습니다.")
        event_source = stream_agent
        if event_source is None and call_agent is not None:

            def _single_result_stream(user_input: str) -> Iterable[AgentStreamEvent]:
                yield AgentStreamEvent(
                    event="final_response",
                    data={},
                    result=call_agent(user_input),
                )

            event_source = _single_result_stream

        if event_source is None:
            raise ValueError("stream_agent or call_agent must be provided")

        for event in event_source(prompt):
            if event.event == "final_response" and event.result is not None:
                result = event.result
                display_answer = _clean_saved_file_notice(result.answer, result.file_path or "")
                status_placeholder.markdown(display_answer or _DEFAULT_ERROR_MESSAGE)
                continue
            if event.event == "error":
                last_error_message = str(
                    event.data.get("message")
                    or "응답 처리 중 오류가 발생했습니다."
                )
                status_placeholder.markdown(last_error_message)
                continue

            progress_message = _progress_message_for_event(event)
            if progress_message:
                status_placeholder.markdown(progress_message)

    if result is None:
        result = AgentCallResult(answer=last_error_message)

    append_assistant_message(
        {
            "role": "assistant",
            "content": _clean_saved_file_notice(result.answer, result.file_path or ""),
            "file_path": result.file_path or "",
            "evidence": result.evidence_items,
        }
    )
    st.rerun()


def _progress_message_for_event(event: AgentStreamEvent) -> str:
    if event.event == "request_started":
        return "요청을 접수했습니다."
    if event.event == "progress_snapshot":
        summary = str(event.data.get("summary") or "").strip()
        return summary or _DEFAULT_PROCESSING_MESSAGE
    if event.event not in {"stage_started", "heartbeat"}:
        return ""
    stage = str(event.data.get("stage") or "").strip()
    return _STAGE_MESSAGES.get(stage, _DEFAULT_PROCESSING_MESSAGE)


def _render_evidence(evidence_items: list[object]) -> None:
    with st.expander("근거 보기"):
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "") or "").strip()
            source = str(item.get("url_or_path", "") or "").strip()
            title = str(item.get("title", "") or "").strip()
            if title:
                st.markdown(f"- `{kind}`: **{title}** ({source})")
            else:
                st.markdown(f"- `{kind}`: {source}")


def _render_download_button(file_path: str, fastapi_url: str) -> None:
    filename = os.path.basename(file_path)
    download_url = f"{fastapi_url}/download/{quote(filename)}"
    safe_filename = escape(filename)
    safe_download_url = escape(download_url, quote=True)

    st.markdown(
        f'<div class="dm-save-note">파일 저장 완료: <code>{safe_filename}</code></div>'
        f'<a href="{safe_download_url}" target="_blank" download="{safe_filename}">'
        f'<button class="dm-download-button">'
        f"파일 다운로드 ({safe_filename})"
        f"</button></a>",
        unsafe_allow_html=True,
    )


def _clean_saved_file_notice(content: str, file_path: str | None) -> str:
    if not file_path:
        return content

    cleaned_lines = [
        line
        for line in content.splitlines()
        if not re.match(r"^\s*저장 완료\s*:", line)
    ]
    return "\n".join(cleaned_lines).strip() or content
