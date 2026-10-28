"""Browser-tab persistence, native chat drafts and sidebar focus recovery.

Streamlit's chat input only sends text on submit. The bridge observes user
navigation, while restoration uses Streamlit's supported session-state prefill.
"""
from pathlib import Path

import streamlit as st
from pydantic import BaseModel, ValidationError
from streamlit.components.v2 import component

from src.app.web.conversation_snapshot import dump_conversations, load_conversations
from src.app.web.streamlit_state import Conversation, list_conversations
from src.infra.settings import get_settings

_KEY = "documate_browser_state"
_bridge = component(
    "documate_browser_state",
    js=Path(__file__).with_name("streamlit_browser_state.js").read_text(encoding="utf-8"),
)


class BrowserRestore(BaseModel):
    snapshot: str | None = None
    error: str | None = None


def restore_browser_session() -> bool:
    if st.session_state.get("browser_session_ready"):
        return True
    result = _bridge(
        data={"restore": True}, key=_KEY, on_restore_change=lambda: None,
        on_draft_change=_record_draft, on_storage_error_change=lambda: None, height=0,
    )
    if result.get("restore") is None:
        st.info("이 탭의 대화를 복원하고 있습니다…")
        return False
    try:
        restored = BrowserRestore.model_validate(result["restore"])
        if restored.snapshot is not None:
            conversations, active_id = load_conversations(
                restored.snapshot, fastapi_url=get_settings().fastapi_url,
            )
            st.session_state["conversations"] = conversations
            st.session_state["active_conversation_id"] = active_id
        if restored.error:
            st.session_state["browser_storage_notice"] = restored.error
    except ValidationError:
        st.session_state["browser_storage_notice"] = "저장된 대화를 복원하지 못했습니다. 새 대화로 시작합니다."
    st.session_state["browser_session_ready"] = True
    # The bootstrap and normal bridge share one widget key, on separate runs.
    st.rerun()


def _record_draft() -> None:
    event = st.session_state.get(_KEY, {}).get("draft")
    if not isinstance(event, dict):
        return
    conversation = st.session_state.get("conversations", {}).get(event.get("conversation_id"))
    if conversation is None or not isinstance(event.get("text"), str) or type(event.get("revision")) is not int:
        return
    conversation.draft.record(event["text"], revision=event["revision"])


def render_browser_state(conversation: Conversation) -> None:
    state = st.session_state.get(_KEY, {})
    result = _bridge(
        data={"conversation_id": conversation.id, "revision": conversation.draft.revision,
              "text": conversation.draft.text,
              "focus_input": conversation.draft.needs_restore,
              "pending_chat": conversation.pending_request is not None and conversation.pending_request.kind == "chat",
              "message_count": len(conversation.messages),
              "max_file_bytes": get_settings().upload_max_file_mib * 1024 * 1024,
              "snapshot": dump_conversations(list_conversations(), conversation.id),
              "storage_error": state.get("storage_error")},
        key=_KEY,
        on_draft_change=_record_draft, on_restore_change=lambda: None,
        on_storage_error_change=lambda: None,
        height=0,
    )
    notice = result.get("storage_error") or st.session_state.get("browser_storage_notice")
    if notice:
        st.warning(notice)


def restore_composer(conversation: Conversation) -> None:
    if conversation.draft.needs_restore:
        st.session_state[conversation.widget_key("chat_input")] = conversation.draft.text
        conversation.draft.needs_restore = False


def edit_question(conversation: Conversation, prompt: str) -> None:
    if not conversation.is_read_only and not conversation.is_busy and conversation.pending_upload is None:
        conversation.draft.edit(prompt)
