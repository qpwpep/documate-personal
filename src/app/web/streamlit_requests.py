"""Consume UI submissions once, before rendering the in-flight screen."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.app.web.streamlit_state import Conversation


@dataclass(frozen=True)
class ChatSubmission:
    prompt: str
    kind: Literal["chat"] = field(default="chat", init=False)


@dataclass(frozen=True)
class SaveSubmission:
    answer_id: str
    kind: Literal["save"] = field(default="save", init=False)


@dataclass(frozen=True)
class UploadSubmission:
    kind: Literal["upload"] = field(default="upload", init=False)


def queue_chat(conversation: Conversation, prompt: str) -> None:
    if conversation.is_read_only or conversation.pending_request is not None or conversation.operation_in_progress:
        return
    conversation.pending_request = ChatSubmission(prompt)


def send_saved_prompt(conversation: Conversation) -> None:
    if not conversation.saved_prompt:
        return
    queue_chat(conversation, conversation.saved_prompt)
    if conversation.pending_request is not None and conversation.pending_request.kind == "chat" and conversation.pending_request.prompt == conversation.saved_prompt:
        conversation.saved_prompt = None


def discard_saved_prompt(conversation: Conversation) -> None:
    conversation.saved_prompt = None


def edit_saved_prompt(conversation: Conversation) -> None:
    if conversation.is_read_only or conversation.is_busy or conversation.pending_upload is not None:
        return
    if conversation.saved_prompt is not None:
        conversation.draft.edit(conversation.saved_prompt)
        conversation.saved_prompt = None


def queue_save(conversation: Conversation, answer_id: str) -> None:
    if conversation.is_read_only or conversation.pending_request is not None or conversation.operation_in_progress:
        return
    conversation.pending_request = SaveSubmission(answer_id)


def queue_upload(conversation: Conversation) -> None:
    if conversation.is_read_only or conversation.pending_request is not None or conversation.operation_in_progress:
        return
    conversation.pending_request = UploadSubmission()
