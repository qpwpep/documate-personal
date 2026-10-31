"""Consume UI submissions once, before rendering the in-flight screen."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from src.app.web.streamlit_state import Conversation


@dataclass(frozen=True)
class ChatSubmission:
    prompt: str
    kind: Literal["chat"] = field(default="chat", init=False)
    operation_id: str = field(default_factory=lambda: str(uuid4()), init=False)


@dataclass(frozen=True)
class SaveSubmission:
    answer_id: str
    kind: Literal["save"] = field(default="save", init=False)
    operation_id: str = field(default_factory=lambda: str(uuid4()), init=False)


@dataclass(frozen=True)
class UploadSubmission:
    kind: Literal["upload"] = field(default="upload", init=False)
    operation_id: str = field(default_factory=lambda: str(uuid4()), init=False)


def queue_chat(conversation: Conversation, prompt: str) -> None:
    if conversation.is_read_only or conversation.is_busy:
        return
    conversation.pending_request = ChatSubmission(prompt)


def interrupt_operation(conversation: Conversation, operation_id: str) -> bool:
    submission = conversation.active_submission
    if submission is None or submission.operation_id != operation_id:
        return False
    # Fast reruns may render the next script before the old HTTP stream exits.
    # Release UI state here, once, rather than waiting for its exception handler.
    conversation.active_submission = None
    conversation.operation_in_progress = False
    conversation.draft.needs_restore = True
    if conversation.chat_task is not None:
        conversation.chat_task.stopped.set()
        conversation.chat_task = None
        conversation.client.manifest = None
    if submission.kind == "chat":
        if len(conversation.messages) == conversation.operation_message_count:
            conversation.messages.append({"role": "user", "content": submission.prompt})
        if len(conversation.messages) == conversation.operation_message_count + 1:
            conversation.messages.append({"role": "assistant", "result": None, "interrupted": True})
    elif submission.kind == "save":
        for message in conversation.messages:
            if (message["role"] == "assistant" and message["result"] is not None
                    and message["result"].answer_id == submission.answer_id):
                message["export_error"] = "저장 결과 수신이 중단되었습니다. 같은 답변의 ‘저장 다시 시도’로 결과를 확인할 수 있습니다."
    elif conversation.pending_upload is not None:
        conversation.pending_upload.requires_confirmation = True
        conversation.pending_upload.had_failure = True
        conversation.pending_upload.error = "자료 준비 결과 수신이 중단되었습니다. 첨부 상태를 확인한 뒤 다시 시도해 주세요."
    return True


def get_upload_followup_prompt(conversation: Conversation) -> str | None:
    """Offer only the latest question explicitly blocked by missing attachments."""
    if (conversation.is_read_only or conversation.is_busy or conversation.pending_upload is not None
            or conversation.saved_prompt or conversation.client.manifest is None
            or not conversation.client.manifest.files or len(conversation.messages) < 2):
        return None
    question, answer = conversation.messages[-2:]
    if question["role"] != "user" or answer["role"] != "assistant":
        return None
    result = answer["result"]
    if result is None or result.status != "needs_input" or "upload" not in result.missing_slots:
        return None
    return question["content"]


def continue_upload_question(conversation: Conversation) -> None:
    prompt = get_upload_followup_prompt(conversation)
    if prompt is not None:
        queue_chat(conversation, prompt)


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
    if conversation.is_read_only or conversation.is_busy:
        return
    conversation.pending_request = SaveSubmission(answer_id)


def queue_upload(conversation: Conversation) -> None:
    if conversation.is_read_only or conversation.is_busy:
        return
    conversation.pending_request = UploadSubmission()
