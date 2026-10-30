from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

import streamlit as st

from src.infra.logging_utils import log_event
from src.infra.runtime_paths import get_uploads_dir
from src.infra.settings import get_settings
from src.app.client import AgentRequestContext, AgentSessionClient, UploadAPIError
from src.app.uploads import StagedUpload
from src.app.web.streamlit_theme import THEME_STATE_KEY
from src.app.web.composer_draft import ComposerDraft
from src.core.contracts.outcome import AnswerExportResult, TurnResult
from src.core.uploads import UploadManifest, UploadSyncRequest

if TYPE_CHECKING:
    from src.app.web.chat_response_task import ChatResponseTask
    from src.app.web.streamlit_requests import ChatSubmission, SaveSubmission, UploadSubmission


_THEME_QUERY_MAP = {
    "system": "시스템",
    "light": "라이트",
    "dark": "다크",
    "시스템": "시스템",
    "라이트": "라이트",
    "다크": "다크",
}


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
    error_detail: str | None = None
    attempted: bool = False
    had_failure: bool = False
    needs_refresh_review: bool = False
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        self.base_manifest = self.base_manifest.model_copy(deep=True)


class UserChatMessage(TypedDict):
    role: Literal["user"]
    content: str


class AssistantChatMessage(TypedDict):
    role: Literal["assistant"]
    result: TurnResult | None
    transport_errors: NotRequired[list[str]]
    request_id: NotRequired[str]
    export_result: NotRequired[AnswerExportResult]
    export_error: NotRequired[str]
    interrupted: NotRequired[bool]


ChatMessage = UserChatMessage | AssistantChatMessage


@dataclass
class Conversation:
    """One browser-session conversation; server state remains subject to its TTL."""

    id: str
    client: AgentSessionClient
    messages: list[ChatMessage] = field(default_factory=list)
    pending_upload: PendingUpload | None = None
    saved_prompt: str | None = None
    draft: ComposerDraft = field(default_factory=ComposerDraft)
    followup_prompt: str | None = None
    slack_drafts: dict[str, str] = field(default_factory=dict)
    pending_request: ChatSubmission | SaveSubmission | UploadSubmission | None = None
    active_submission: ChatSubmission | SaveSubmission | UploadSubmission | None = None
    operation_message_count: int = 0
    operation_in_progress: bool = False
    chat_task: ChatResponseTask | None = None
    confirmed_epoch: str | None = None
    preserved_manifest: UploadManifest | None = None
    expired: bool = False
    validation_error: str | None = None

    @property
    def is_read_only(self) -> bool:
        return self.expired or self.validation_error is not None

    @property
    def is_busy(self) -> bool:
        return self.operation_in_progress or self.pending_request is not None

    @property
    def has_interrupted_response(self) -> bool:
        return bool(self.messages and self.messages[-1]["role"] == "assistant"
                    and self.messages[-1].get("interrupted"))

    @property
    def title(self) -> str:
        for message in self.messages:
            if message["role"] == "user":
                text = " ".join(message["content"].split())
                return text[:45] + ("…" if len(text) > 45 else "")
        return "새 대화"

    def widget_key(self, name: str) -> str:
        return f"conversation:{self.id}:{name}"


def ensure_theme_state() -> None:
    """Seed the preference once; later reruns keep the user's radio selection."""
    if THEME_STATE_KEY not in st.session_state:
        raw_theme = st.query_params.get("theme", "")
        st.session_state[THEME_STATE_KEY] = _THEME_QUERY_MAP.get(
            raw_theme.strip().lower(), "시스템"
        )


def ensure_session_state(logger: logging.Logger) -> None:
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {}
        _start_new_session(logger, "streamlit_session_start")
    get_session_path().mkdir(parents=True, exist_ok=True)


def get_conversation() -> Conversation:
    return st.session_state["conversations"][st.session_state["active_conversation_id"]]


def list_conversations() -> list[Conversation]:
    return list(st.session_state["conversations"].values())


def get_session_id() -> str:
    return get_conversation().id


def get_session_path() -> Path:
    session_path = get_uploads_dir() / get_session_id()
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path


def get_session_client() -> AgentSessionClient:
    return get_conversation().client


def get_upload_manifest() -> UploadManifest | None:
    """Read-only conversations expose the last snapshot for display, never reuse."""
    conversation = get_conversation()
    if conversation.is_read_only:
        return conversation.preserved_manifest
    _remember_manifest(conversation)
    if conversation.has_interrupted_response and conversation.client.manifest is None:
        # Recovery must not wait on the still-running server request's lock.
        # This snapshot is display-only; explicit submissions validate again.
        return conversation.preserved_manifest
    return conversation.preserved_manifest if conversation.is_read_only else conversation.client.manifest


def get_pending_upload() -> PendingUpload | None:
    return get_conversation().pending_upload


def set_pending_upload(operation: PendingUpload | None) -> None:
    get_conversation().pending_upload = operation


def get_messages() -> list[ChatMessage]:
    return get_conversation().messages


def append_message(message: ChatMessage) -> None:
    get_messages().append(message)


def reset_chat_session(logger: logging.Logger) -> None:
    if get_conversation().is_busy and get_conversation().chat_task is None:
        return
    _pause_automatic_work(get_conversation())
    _remember_manifest(get_conversation())
    _start_new_session(logger, "streamlit_session_reset")
    get_session_path().mkdir(parents=True, exist_ok=True)


def switch_conversation(conversation_id: str) -> bool:
    current = get_conversation()
    if (current.is_busy and current.chat_task is None) or current.id == conversation_id:
        return False
    target = st.session_state["conversations"][conversation_id]
    if target.is_busy and target.chat_task is None:
        return False
    _pause_automatic_work(current)
    _remember_manifest(current)
    _pause_automatic_work(target)
    st.session_state["active_conversation_id"] = target.id
    target.draft.needs_restore = True
    if not target.has_interrupted_response and not target.is_busy:
        validate_conversation(target)
    return True


def validate_conversation(
    conversation: Conversation | None = None, *,
    expected_submission: ChatSubmission | SaveSubmission | UploadSubmission | None = None,
) -> bool:
    """Recheck server identity without replaying uploads, questions or actions."""
    conversation = conversation or get_conversation()
    if conversation.expired:
        return False
    _remember_manifest(conversation)
    if conversation.expired:
        return False
    expected_epoch = conversation.confirmed_epoch
    try:
        manifest = conversation.client.refresh_uploads()
    except UploadAPIError as exc:
        if expected_submission is not None and conversation.active_submission is not expected_submission:
            return False
        conversation.validation_error = str(exc)
        return False
    if expected_submission is not None and conversation.active_submission is not expected_submission:
        return False
    if expected_epoch is not None and manifest.epoch != expected_epoch:
        conversation.expired = True
        conversation.validation_error = None
        conversation.client.manifest = None
        _pause_automatic_work(conversation)
        return False
    conversation.validation_error = None
    _remember_manifest(conversation)
    return True


def _remember_manifest(conversation: Conversation) -> None:
    if conversation.is_read_only:
        return
    manifest = conversation.client.manifest
    if manifest is not None:
        if conversation.confirmed_epoch is not None and manifest.epoch != conversation.confirmed_epoch:
            conversation.expired = True
            conversation.client.manifest = None
            _pause_automatic_work(conversation)
            return
        conversation.preserved_manifest = manifest.model_copy(deep=True)
        if conversation.confirmed_epoch is None:
            conversation.confirmed_epoch = manifest.epoch


def _pause_automatic_work(conversation: Conversation) -> None:
    if conversation.followup_prompt:
        conversation.saved_prompt = conversation.followup_prompt
        conversation.followup_prompt = None
    if conversation.pending_upload is not None:
        conversation.pending_upload.requires_confirmation = True


def _start_new_session(logger: logging.Logger, event_name: str) -> None:
    session_id = str(uuid.uuid4())
    conversation = Conversation(
        id=session_id,
        client=AgentSessionClient(AgentRequestContext(
            fastapi_url=get_settings().fastapi_url, session_id=session_id,
        )),
    )
    st.session_state["conversations"][session_id] = conversation
    st.session_state["active_conversation_id"] = session_id
    log_event(logger, logging.INFO, event_name, session_id=session_id[:8])
