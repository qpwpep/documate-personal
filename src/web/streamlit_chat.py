from __future__ import annotations

import os
from collections.abc import Callable

import streamlit as st

from src.web.streamlit_api_client import AgentCallResult
from src.web.streamlit_state import ChatMessage


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
    call_agent: Callable[[str], AgentCallResult],
    prompt: str,
    append_user_message: Callable[[ChatMessage], None],
    append_assistant_message: Callable[[ChatMessage], None],
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

    with st.spinner("Agent가 생각 중입니다..."):
        result = call_agent(prompt)

    append_assistant_message(
        {
            "role": "assistant",
            "content": result.answer,
            "file_path": result.file_path or "",
            "evidence": result.evidence_items,
        }
    )
    st.rerun()


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
    st.info(f"💾 **파일 저장 완료:** `{filename}`")
    st.markdown(
        f'<a href="{download_url}" target="_blank" download="{filename}">'
        f'<button style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; width: 100%;">'
        f"⬇️ 파일 다운로드 ({filename})"
        f"</button></a>",
        unsafe_allow_html=True,
    )
