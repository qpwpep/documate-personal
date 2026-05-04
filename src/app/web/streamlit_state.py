from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, TypedDict

import streamlit as st

from src.infra.logging_utils import log_event
from src.infra.runtime_paths import get_uploads_dir


QUICK_PROMPTS_STATE_KEY = "documate_quick_prompts"


class ChatMessage(TypedDict):
    role: str
    content: str
    file_path: str
    evidence: list[Any]


def ensure_session_state(logger: logging.Logger) -> None:
    if "session_id" not in st.session_state:
        _start_new_session(logger, "streamlit_session_start")

    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = None

    if "messages" not in st.session_state:
        st.session_state["messages"] = [_build_default_assistant_message()]

    get_session_path().mkdir(parents=True, exist_ok=True)


def get_session_id() -> str:
    return str(st.session_state["session_id"])


def get_session_path() -> Path:
    session_path = get_uploads_dir() / get_session_id()
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path


def get_uploaded_file_name() -> str | None:
    file_name = st.session_state.get("uploaded_file_name")
    return str(file_name) if file_name else None


def set_uploaded_file_name(file_name: str | None) -> None:
    st.session_state["uploaded_file_name"] = file_name


def clear_uploaded_file_name() -> None:
    st.session_state["uploaded_file_name"] = None


def get_messages() -> list[ChatMessage]:
    return st.session_state["messages"]


def append_message(message: ChatMessage) -> None:
    get_messages().append(message)


def reset_chat_session(logger: logging.Logger) -> None:
    _start_new_session(logger, "streamlit_session_reset")
    st.session_state["uploaded_file_name"] = None
    st.session_state["messages"] = [_build_default_assistant_message()]
    st.session_state.pop(QUICK_PROMPTS_STATE_KEY, None)
    get_session_path().mkdir(parents=True, exist_ok=True)


def _start_new_session(logger: logging.Logger, event_name: str) -> None:
    session_id = str(uuid.uuid4())
    st.session_state["session_id"] = session_id
    log_event(logger, logging.INFO, event_name, session_id=session_id[:8])


def _build_default_assistant_message() -> ChatMessage:
    return {
        "role": "assistant",
        "content": "안녕하세요. 질문을 입력하거나 왼쪽에서 코드 파일을 업로드해 주세요.",
        "file_path": "",
        "evidence": [],
    }
