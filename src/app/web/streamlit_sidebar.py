from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st
from pydantic import ValidationError

from src.app.web.streamlit_theme import THEME_OPTIONS, THEME_STATE_KEY
from src.app.web.streamlit_state import get_conversation
from src.app.web.streamlit_conversations import render_conversation_picker
from src.core.uploads import UploadManifest
from src.core.slack_contract import RecipientInput, RecipientSelector


@dataclass
class SidebarInputs:
    slack_recipient: RecipientInput | None
    new_chat_requested: bool
    selected_conversation: str | None = None
    remove_file_id: str | None = None
    clear_uploads_requested: bool = False
    refresh_uploads_requested: bool = False



def render_sidebar(
    *,
    manifest: UploadManifest | None = None,
    uploads_busy: bool = False,
    interactions_disabled: bool = False,
) -> SidebarInputs:
    conversation = get_conversation()
    uploads_busy = uploads_busy or interactions_disabled
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
            disabled=interactions_disabled,
        )
        selected_conversation = render_conversation_picker(disabled=interactions_disabled)

        st.markdown('<div class="dm-sidebar-section">함께 검색할 파일</div>', unsafe_allow_html=True)
        if manifest is not None:
            refresh_uploads_requested = st.button("첨부 목록 새로고침", key="documate_refresh_uploads", disabled=uploads_busy)
        if manifest is not None and not refresh_uploads_requested:
            total_mib = sum(item.size_bytes for item in manifest.files) / (1024 * 1024)
            state_label = "이전 첨부 기록" if conversation.is_read_only else "검색 가능"
            st.markdown(f'<p class="dm-upload-summary"><strong>{state_label} {len(manifest.files)}개 · {total_mib:.2f} MiB</strong></p>', unsafe_allow_html=True)
            for item in manifest.files:
                availability = "이전 첨부 기록" if conversation.is_read_only else "검색 가능"
                st.markdown(f'<div class="dm-upload-note"><strong>{escape(item.name)}</strong> · {item.size_bytes / 1024:.1f} KiB · {availability}</div>', unsafe_allow_html=True)
                if st.button(f"{item.name} 삭제", key=f"documate_remove_upload_{item.file_id}", disabled=uploads_busy):
                    remove_file_id = item.file_id
            if manifest.files:
                clear_uploads_requested = st.button("전체 첨부 해제", key="documate_clear_uploads", disabled=uploads_busy)
                st.markdown('<p class="dm-upload-help">첨부를 해제해도 대화와 기존 답변의 인용은 유지됩니다.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="dm-upload-help">검색 준비가 끝난 파일이 없습니다. 입력창에서 선택한 파일은 전송 후 여기에 표시됩니다.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="dm-upload-note">첨부 상태를 확인하지 못했습니다. 서버 목록을 확인한 뒤 질문할 수 있습니다.</p>', unsafe_allow_html=True)

        st.markdown('<div class="dm-sidebar-section">화면 모드</div>', unsafe_allow_html=True)
        st.radio(
            "테마",
            options=THEME_OPTIONS,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key=THEME_STATE_KEY,
            disabled=interactions_disabled,
        )

        st.markdown('<div class="dm-sidebar-section">Slack 전송</div>', unsafe_allow_html=True)
        st.caption('수신자를 설정한 뒤 질문 입력창에 "방금 답변을 Slack으로 보내줘"라고 요청하세요. 설정만으로 전송하지 않습니다.')
        recipient_options = ("미지정", "채널", "사용자", "이메일")
        recipient_kind = st.radio(
            "수신자 유형", options=recipient_options,
            index=recipient_options.index(conversation.slack_drafts.get("kind", "미지정")),
            key=conversation.widget_key("slack_recipient_kind"),
            disabled=interactions_disabled or conversation.is_read_only,
        )
        conversation.slack_drafts["kind"] = recipient_kind
        slack_recipient = None
        if recipient_kind != "미지정":
            kinds = {
                "채널": ("channel", "C0123456789", "채널 이름 대신 C, G 또는 D로 시작하는 채널 ID를 입력하세요."),
                "사용자": ("user", "U0123456789", "표시 이름 대신 U 또는 W로 시작하는 멤버 ID를 입력하세요. 이메일로도 지정할 수 있습니다."),
                "이메일": ("email", "name@example.com", "받는 사람이 Slack 워크스페이스에 등록한 이메일을 입력하세요."),
            }
            kind, placeholder, guidance = kinds[recipient_kind]
            st.caption(guidance)
            value = st.text_input("Slack 수신자", value=conversation.slack_drafts.get(kind, ""), placeholder=placeholder,
                                  key=conversation.widget_key(f"slack_recipient_{kind}"),
                                  disabled=interactions_disabled or conversation.is_read_only)
            conversation.slack_drafts[kind] = value
            slack_recipient = RecipientInput(kind=kind, value=value)
            try:
                RecipientSelector(kind=kind, value=value)
            except ValidationError:
                st.caption("수신자 입력을 확인해 주세요. 일반 질문은 계속할 수 있고, Slack 전송은 올바른 수신자가 필요합니다.")
            if recipient_kind in {"채널", "사용자"}:
                with st.expander("수신자 ID 찾는 방법", expanded=False):
                    if recipient_kind == "채널":
                        st.markdown("Slack에서 대상 채널을 열고 상단 채널 이름을 누르세요. 채널 상세 화면 맨 아래의 **채널 ID**를 복사해 입력하세요.")
                    else:
                        st.markdown("Slack에서 받는 사람의 프로필을 열고 **더 보기 → 멤버 ID 복사**를 선택하세요. ID를 찾기 어렵다면 수신자 유형을 **이메일**로 바꿀 수 있습니다.")

    return SidebarInputs(
        slack_recipient=slack_recipient,
        new_chat_requested=new_chat_requested,
        selected_conversation=selected_conversation,
        remove_file_id=remove_file_id,
        clear_uploads_requested=clear_uploads_requested,
        refresh_uploads_requested=refresh_uploads_requested,
    )

