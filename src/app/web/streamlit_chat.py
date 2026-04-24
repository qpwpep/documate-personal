from __future__ import annotations

import os
from collections.abc import Callable, Iterable

import streamlit as st

from src.app.web.streamlit_api_client import AgentCallResult, AgentStreamEvent
from src.app.web.streamlit_state import ChatMessage


_STAGE_MESSAGES = {
    "summarize": "대화 맥락 정리 중...",
    "planner": "질문 분석 중...",
    "retrieval": "근거 수집 중...",
    "pre_synthesis_validation": "검색 결과 확인 중...",
    "synthesis": "응답 생성 중...",
    "post_synthesis_validation": "응답 근거 확인 중...",
    "validation": "근거 검증 중...",
    "action_postprocess": "결과 정리 중...",
}


def render_chat_history(messages: list[ChatMessage], fastapi_url: str) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            evidence_items = message.get("evidence") or []
            if message["role"] == "assistant" and evidence_items:
                _render_evidence(evidence_items)

            file_path = message.get("file_path", "")
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
    last_error_message = "응답을 받지 못했습니다."
    with st.chat_message("assistant"):
        status_placeholder = st.empty() if hasattr(st, "empty") else st
        status_placeholder.markdown("요청 접수 중...")
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
                status_placeholder.markdown(result.answer or "응답을 받지 못했습니다.")
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
            "content": result.answer,
            "file_path": result.file_path or "",
            "evidence": result.evidence_items,
        }
    )
    st.rerun()


def _progress_message_for_event(event: AgentStreamEvent) -> str:
    if event.event == "request_started":
        return "요청 접수 중..."
    if event.event not in {"stage_started", "heartbeat"}:
        return ""
    stage = str(event.data.get("stage") or "").strip()
    return _STAGE_MESSAGES.get(stage, "응답 준비 중...")


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
    download_url = f"{fastapi_url}/download/{filename}"

    st.markdown("---")
    st.info(f"📄 **파일 저장 완료:** `{filename}`")
    st.markdown(
        f'<a href="{download_url}" target="_blank" download="{filename}">'
        f'<button style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; width: 100%;">'
        f"파일 다운로드 ({filename})"
        f"</button></a>",
        unsafe_allow_html=True,
    )
