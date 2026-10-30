"""Browser-session navigation and explicit server-session recovery."""
from __future__ import annotations

import streamlit as st

from src.app.web.streamlit_state import (
    Conversation,
    get_conversation,
    list_conversations,
    validate_conversation,
)


def render_conversation_picker(*, disabled: bool = False) -> str | None:
    current = get_conversation()
    conversations = list_conversations()
    by_id = {conversation.id: conversation for conversation in conversations}
    ids = list(reversed(by_id))
    with st.sidebar:
        recent_ids = ids[:3]
        if current.id not in recent_ids:
            recent_ids.append(current.id)
        with st.container(key="documate_recent_conversations"):
            st.caption("최근 대화")
            for conversation_id in recent_ids:
                is_current = conversation_id == current.id
                label = _conversation_label(by_id[conversation_id], conversations)
                if st.button(
                    f"현재 · {label}" if is_current else label,
                    key=current.widget_key(f"recent:{conversation_id}"),
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                    disabled=disabled or is_current,
                ):
                    return conversation_id
        with st.expander("모든 대화 보기", expanded=False):
            selected = st.selectbox(
                "대화 선택", ids, index=ids.index(current.id),
                format_func=lambda value: _conversation_label(by_id[value], conversations),
                key=current.widget_key(f"picker:{current.title}:{current.expired}"), disabled=disabled,
            )
            st.caption("이 탭에서 새로고침해도 대화와 초안을 복원합니다. 서버 세션이 만료되면 기존 기록은 읽기 전용으로 남습니다.")
    return selected if selected != current.id else None


def render_conversation_notice() -> None:
    conversation = get_conversation()
    if conversation.expired:
        st.warning(
            "서버에서 이 대화가 만료되거나 초기화되어 읽기 전용으로 표시합니다. "
            "기존 메시지와 당시 출처는 확인할 수 있지만 질문·저장·전송·첨부는 다시 실행하지 않습니다. "
            "계속하려면 새 채팅을 시작하고 필요한 파일을 다시 첨부해 주세요."
        )
    elif conversation.validation_error:
        st.warning("대화의 서버 상태를 확인하지 못했습니다. 기록을 유지한 채 연결을 다시 확인해 주세요.")
        st.caption(conversation.validation_error)
        if st.button("대화 연결 다시 확인", key=conversation.widget_key("retry_connection")):
            validate_conversation(conversation)
            st.rerun()
    if conversation.is_read_only:
        drafts = [conversation.draft.text, conversation.saved_prompt]
        if conversation.pending_upload is not None:
            drafts.append(conversation.pending_upload.prompt)
        for prompt in dict.fromkeys(value for value in drafts if value):
            st.info(f"전송하지 않은 질문: {prompt}")


def _conversation_label(conversation: Conversation, conversations: list[Conversation]) -> str:
    index = conversations.index(conversation) + 1
    suffix = " · 읽기 전용" if conversation.expired else ""
    if conversation.chat_task is not None:
        suffix = " · 답변 준비 중"
    return f"{index}. {conversation.title}{suffix}"
