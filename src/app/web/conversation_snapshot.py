"""Validated display state for reconnecting a browser tab to its conversations."""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.app.client import AgentRequestContext, AgentSessionClient
from src.app.web.composer_draft import ComposerDraft
from src.app.web.streamlit_state import ChatMessage, Conversation
from src.core.uploads import UploadManifest


class StoredDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = ""
    revision: int = Field(default=0, ge=0)


class StoredConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    messages: list[ChatMessage]
    draft: StoredDraft
    saved_prompt: str | None = None
    slack_drafts: dict[str, str] = Field(default_factory=dict)
    confirmed_epoch: str | None = None
    preserved_manifest: UploadManifest | None = None
    expired: bool = False


class BrowserSessionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    active_conversation_id: UUID
    conversations: list[StoredConversation] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_selection(self) -> BrowserSessionSnapshot:
        ids = [conversation.id for conversation in self.conversations]
        if len(set(ids)) != len(ids) or self.active_conversation_id not in ids:
            raise ValueError("the selected conversation must belong to a unique conversation list")
        return self


def dump_conversations(conversations: list[Conversation], active_id: str) -> str:
    stored = []
    for conversation in conversations:
        # Retain unsent text, never a command that could repeat an upload,
        # question, save or Slack action when the page reconnects.
        saved_prompt = conversation.saved_prompt or conversation.followup_prompt
        if conversation.pending_upload is not None:
            saved_prompt = saved_prompt or conversation.pending_upload.prompt
        stored.append(StoredConversation(
            id=UUID(conversation.id), messages=conversation.messages,
            draft=StoredDraft(text=conversation.draft.text, revision=conversation.draft.revision),
            saved_prompt=saved_prompt, slack_drafts=conversation.slack_drafts,
            confirmed_epoch=conversation.confirmed_epoch,
            preserved_manifest=conversation.preserved_manifest,
            expired=conversation.expired,
        ))
    return BrowserSessionSnapshot(
        active_conversation_id=UUID(active_id), conversations=stored,
    ).model_dump_json()


def load_conversations(snapshot: str, *, fastapi_url: str) -> tuple[dict[str, Conversation], str]:
    saved = BrowserSessionSnapshot.model_validate_json(snapshot)
    conversations = {}
    for item in saved.conversations:
        conversation_id = str(item.id)
        conversations[conversation_id] = Conversation(
            id=conversation_id,
            # A cached manifest is display-only until the server confirms its epoch.
            client=AgentSessionClient(AgentRequestContext(
                fastapi_url=fastapi_url, session_id=conversation_id,
            )),
            messages=item.messages,
            draft=ComposerDraft(text=item.draft.text, revision=item.draft.revision, needs_restore=True),
            saved_prompt=item.saved_prompt, slack_drafts=item.slack_drafts,
            confirmed_epoch=item.confirmed_epoch, preserved_manifest=item.preserved_manifest,
            expired=item.expired,
        )
    return conversations, str(saved.active_conversation_id)
