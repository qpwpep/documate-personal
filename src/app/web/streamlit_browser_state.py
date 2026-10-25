"""Small browser bridge for native chat drafts and sidebar focus recovery.

Streamlit's chat input only sends text on submit. The bridge observes user
navigation, while restoration uses Streamlit's supported session-state prefill.
"""
from pathlib import Path

import streamlit as st
from streamlit.components.v2 import component

from src.app.web.streamlit_state import Conversation

_KEY = "documate_browser_state"
_bridge = component(
    "documate_browser_state",
    js=Path(__file__).with_name("streamlit_browser_state.js").read_text(encoding="utf-8"),
)


def _record_draft() -> None:
    event = st.session_state.get(_KEY, {}).get("draft")
    if not isinstance(event, dict):
        return
    conversation = st.session_state.get("conversations", {}).get(event.get("conversation_id"))
    if conversation is None or not isinstance(event.get("text"), str) or type(event.get("revision")) is not int:
        return
    conversation.draft.record(event["text"], revision=event["revision"])


def render_browser_state(conversation: Conversation) -> None:
    _bridge(
        data={"conversation_id": conversation.id, "revision": conversation.draft.revision,
              "text": conversation.draft.text,
              "focus_input": conversation.draft.needs_restore},
        key=_KEY,
        on_draft_change=_record_draft,
        height=0,
    )


def restore_composer(conversation: Conversation) -> None:
    if conversation.draft.needs_restore:
        st.session_state[conversation.widget_key("chat_input")] = conversation.draft.text
        conversation.draft.needs_restore = False


def edit_question(conversation: Conversation, prompt: str) -> None:
    if not conversation.is_read_only and not conversation.is_busy and conversation.pending_upload is None:
        conversation.draft.edit(prompt)
