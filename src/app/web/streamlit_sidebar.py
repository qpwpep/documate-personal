from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st

from src.app.web.streamlit_theme import _THEME_OPTIONS, _sync_theme_from_query_params


@dataclass
class SidebarInputs:
    slack_user_id: str
    slack_email: str
    slack_channel_id: str
    theme_mode: str
    new_chat_requested: bool



def render_sidebar(current_file_name: str | None = None) -> SidebarInputs:
    _sync_theme_from_query_params()

    with st.sidebar:
        st.markdown(
            """
            <div class="dm-sidebar-brand">
                <span class="dm-mark">DM</span>
                <h2>DocuMate</h2>
                <p>문서를 근거로 답하고, 필요한 결과를 파일이나 Slack으로 이어서 보냅니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        new_chat_requested = st.button(
            "새 채팅",
            key="documate_new_chat",
            use_container_width=True,
        )

        st.markdown('<div class="dm-sidebar-section">현재 파일</div>', unsafe_allow_html=True)
        uploaded_label = escape(current_file_name or "아직 업로드된 파일이 없습니다.")
        st.markdown(
            f'<div class="dm-upload-note">현재 파일: <strong>{uploaded_label}</strong></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="dm-sidebar-section">화면 모드</div>', unsafe_allow_html=True)
        theme_mode = st.radio(
            "테마",
            options=_THEME_OPTIONS,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="documate_theme_mode",
        )

        st.markdown('<div class="dm-sidebar-section">Slack 전송</div>', unsafe_allow_html=True)
        slack_user_id = st.text_input("User ID", value="", placeholder="Uxxxxx")
        slack_email = st.text_input("Email", value="", placeholder="name@example.com")
        slack_channel_id = st.text_input("Channel ID", value="", placeholder="C/G/Dxxxxx")

    return SidebarInputs(
        slack_user_id=slack_user_id,
        slack_email=slack_email,
        slack_channel_id=slack_channel_id,
        theme_mode=theme_mode,
        new_chat_requested=new_chat_requested,
    )

