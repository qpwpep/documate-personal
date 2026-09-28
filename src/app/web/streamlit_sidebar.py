from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st
from pydantic import ValidationError

from src.app.web.streamlit_theme import _THEME_OPTIONS, _sync_theme_from_query_params
from src.core.uploads import UploadManifest
from src.core.slack_contract import RecipientSelector


@dataclass
class SidebarInputs:
    slack_recipient: RecipientSelector | None
    theme_mode: str
    new_chat_requested: bool
    remove_file_id: str | None = None
    clear_uploads_requested: bool = False
    refresh_uploads_requested: bool = False
    slack_recipient_error: str | None = None



def render_sidebar(
    current_file_name: str | None = None,
    *,
    manifest: UploadManifest | None = None,
    uploads_busy: bool = False,
) -> SidebarInputs:
    _sync_theme_from_query_params()
    remove_file_id = None
    clear_uploads_requested = False
    refresh_uploads_requested = False

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

        st.markdown('<div class="dm-sidebar-section">함께 검색할 파일</div>', unsafe_allow_html=True)
        if manifest is not None:
            total_mib = sum(item.size_bytes for item in manifest.files) / (1024 * 1024)
            st.markdown(f'<p class="dm-upload-summary"><strong>{len(manifest.files)}개 파일 · {total_mib:.2f} MiB</strong></p>', unsafe_allow_html=True)
            for item in manifest.files:
                st.markdown(f'<div class="dm-upload-note"><strong>{escape(item.name)}</strong> · {item.size_bytes / 1024:.1f} KiB · 검색 가능</div>', unsafe_allow_html=True)
                if st.button(f"{item.name} 삭제", key=f"documate_remove_upload_{item.file_id}", disabled=uploads_busy):
                    remove_file_id = item.file_id
            if manifest.files:
                clear_uploads_requested = st.button("전체 첨부 해제", key="documate_clear_uploads", disabled=uploads_busy)
                st.markdown('<p class="dm-upload-help">첨부를 해제해도 대화와 기존 답변의 인용은 유지됩니다.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="dm-upload-help">아직 업로드된 파일이 없습니다.</p>', unsafe_allow_html=True)
            refresh_uploads_requested = st.button("첨부 목록 새로고침", key="documate_refresh_uploads", disabled=uploads_busy)
        else:
            uploaded_label = escape(current_file_name or "아직 업로드된 파일이 없습니다.")
            st.markdown(f'<div class="dm-upload-note">현재 파일: <strong>{uploaded_label}</strong></div>', unsafe_allow_html=True)

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
        recipient_kind = st.radio(
            "수신자 유형", options=("미지정", "채널", "사용자", "이메일"), index=0,
            key="documate_slack_recipient_kind",
        )
        slack_recipient = None
        slack_recipient_error = None
        if recipient_kind != "미지정":
            kinds = {"채널": ("channel", "C/G/Dxxxxx"), "사용자": ("user", "Uxxxxx"), "이메일": ("email", "name@example.com")}
            kind, placeholder = kinds[recipient_kind]
            value = st.text_input("Slack 수신자", value="", placeholder=placeholder,
                                  key=f"documate_slack_recipient_{kind}")
            try:
                slack_recipient = RecipientSelector(kind=kind, value=value)
            except ValidationError:
                slack_recipient_error = "선택한 유형에 맞는 Slack 수신자를 입력하거나 ‘미지정’을 선택해 주세요."
                st.error(slack_recipient_error)

    return SidebarInputs(
        slack_recipient=slack_recipient,
        slack_recipient_error=slack_recipient_error,
        theme_mode=theme_mode,
        new_chat_requested=new_chat_requested,
        remove_file_id=remove_file_id,
        clear_uploads_requested=clear_uploads_requested,
        refresh_uploads_requested=refresh_uploads_requested,
    )

