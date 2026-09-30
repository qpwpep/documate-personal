from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import streamlit as st

from src.infra.logging_utils import log_event
from src.infra.runtime_paths import get_uploads_dir
from src.infra.settings import get_settings
from src.app.client import AgentRequestContext, AgentSessionClient
from src.app.uploads import StagedUpload
from src.core.answer_schema import AnswerResponse, finalize_answer, text_document
from src.core.uploads import UploadManifest, UploadSyncRequest


QUICK_PROMPTS_STATE_KEY = "documate_quick_prompts"


@dataclass
class PendingUpload:
    """UI draft and recovery state; a submitted request stays fixed for retries."""

    base_manifest: UploadManifest
    files: list[StagedUpload] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    clear: bool = False
    replace_file_ids: set[str] = field(default_factory=set)
    request: UploadSyncRequest | None = None
    prompt: str | None = None
    error: str | None = None
    attempted: bool = False
    had_failure: bool = False
    needs_refresh_review: bool = False

    def __post_init__(self) -> None:
        self.base_manifest = self.base_manifest.model_copy(deep=True)


class UserChatMessage(TypedDict):
    role: Literal["user"]
    content: str


class AssistantChatMessage(TypedDict):
    role: Literal["assistant"]
    response: AnswerResponse
    error_messages: NotRequired[list[str]]


ChatMessage = UserChatMessage | AssistantChatMessage


def ensure_session_state(logger: logging.Logger) -> None:
    if "session_id" not in st.session_state:
        _start_new_session(logger, "streamlit_session_start")

    st.session_state.setdefault("pending_upload", None)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [_build_default_assistant_message()]

    get_session_path().mkdir(parents=True, exist_ok=True)


def get_session_id() -> str:
    return str(st.session_state["session_id"])


def get_session_path() -> Path:
    session_path = get_uploads_dir() / get_session_id()
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path


def get_session_client() -> AgentSessionClient:
    return st.session_state["session_client"]


def get_upload_manifest() -> UploadManifest | None:
    """Return the confirmed snapshot; None requires server confirmation, not an empty file set."""
    return get_session_client().manifest


def get_pending_upload() -> PendingUpload | None:
    return st.session_state.get("pending_upload")


def set_pending_upload(operation: PendingUpload | None) -> None:
    st.session_state["pending_upload"] = operation


def get_messages() -> list[ChatMessage]:
    return st.session_state["messages"]


def append_message(message: ChatMessage) -> None:
    get_messages().append(message)


def reset_chat_session(logger: logging.Logger) -> None:
    _start_new_session(logger, "streamlit_session_reset")
    st.session_state["pending_upload"] = None
    st.session_state["messages"] = [_build_default_assistant_message()]
    st.session_state.pop(QUICK_PROMPTS_STATE_KEY, None)
    st.session_state.pop("upload_saved_prompt", None)
    st.session_state.pop("upload_followup_prompt", None)
    get_session_path().mkdir(parents=True, exist_ok=True)


def _start_new_session(logger: logging.Logger, event_name: str) -> None:
    session_id = str(uuid.uuid4())
    st.session_state["session_id"] = session_id
    st.session_state["session_client"] = AgentSessionClient(AgentRequestContext(
        fastapi_url=get_settings().fastapi_url, session_id=session_id,
    ))
    log_event(logger, logging.INFO, event_name, session_id=session_id[:8])


def _build_default_assistant_message() -> ChatMessage:
    return {
        "role": "assistant",
        "response": finalize_answer(
            text_document("안녕하세요. 공식 문서에 대해 질문하거나 코드 파일을 첨부해 주세요."),
            [],
        ),
    }
