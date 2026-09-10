from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4
from langchain_core.messages import HumanMessage
from src.core.answer_schema import AnswerResponse
from src.core.request_contracts import UserTurnSnapshot, required_contract_turn_ids
from src.core.conversation_memory import extract_memory_text

from src.core.contracts import SessionMetadata
from src.core.contracts.graph_state import PendingAction
from src.core.contracts.boundary.runtime import parse_session_metadata
from src.infra.logging_utils import log_event
from src.infra.tools.local_rag import UploadedRetrieverHandle


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConversationMemorySnapshot:
    messages: tuple[Any, ...] = ()
    memory_summary: str | None = None
    user_turns: tuple[UserTurnSnapshot, ...] = ()


def _clone_messages(messages: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(copy.deepcopy(message) for message in messages)


class SessionContext:
    def __init__(self) -> None:
        self._conversation_memory = ConversationMemorySnapshot()
        self.session_metadata: SessionMetadata = parse_session_metadata(None)
        self.upload_retriever_handle: UploadedRetrieverHandle | None = None
        self.upload_file_path: str | None = None
        self.upload_content_hash: str | None = None
        self.previous_response: AnswerResponse | None = None
        self.pending_action: PendingAction | None = None

    @property
    def messages(self) -> list[Any]:
        return list(_clone_messages(self._conversation_memory.messages))

    @messages.setter
    def messages(self, value: Iterable[Any]) -> None:
        self.commit_conversation_memory(
            messages=value,
            memory_summary=self._conversation_memory.memory_summary,
            user_turns=self._conversation_memory.user_turns,
        )

    @property
    def memory_summary(self) -> str | None:
        return self._conversation_memory.memory_summary

    @memory_summary.setter
    def memory_summary(self, value: str | None) -> None:
        self.commit_conversation_memory(
            messages=self._conversation_memory.messages,
            memory_summary=value,
            user_turns=self._conversation_memory.user_turns,
        )

    def snapshot_conversation_memory(self) -> ConversationMemorySnapshot:
        return ConversationMemorySnapshot(
            messages=_clone_messages(self._conversation_memory.messages),
            memory_summary=self._conversation_memory.memory_summary,
            user_turns=self._conversation_memory.user_turns,
        )

    def commit_conversation_memory(
        self,
        *,
        messages: Iterable[Any],
        memory_summary: str | None,
        user_turns: Iterable[UserTurnSnapshot] | None = None,
        preserve_turn_ids: Iterable[str] | None = None,
    ) -> None:
        normalized_summary = str(memory_summary or "").strip() or None
        cloned_messages = list(_clone_messages(messages))
        turns = {turn.turn_id: turn for turn in (
            self._conversation_memory.user_turns if user_turns is None else user_turns
        )}
        for original in self._conversation_memory.user_turns:
            incoming = turns.get(original.turn_id)
            if incoming is not None and incoming != original:
                raise ValueError("original user text cannot change for an existing turn ID")
        for message in cloned_messages:
            if isinstance(message, HumanMessage):
                if not message.id:
                    message.id = f"user:{uuid4().hex}"
                # Existing snapshots contain original text even when dialogue is trimmed.
                if message.id not in turns:
                    turns[message.id] = UserTurnSnapshot(
                        turn_id=message.id, text=extract_memory_text(message.content),
                    )
        if preserve_turn_ids is None:
            preserve_turn_ids = required_contract_turn_ids(self.pending_action.contract) if self.pending_action is not None else ()
        retained_ids = set(list(turns)[-16:]) | set(preserve_turn_ids)
        next_snapshot = ConversationMemorySnapshot(
            messages=tuple(cloned_messages),
            memory_summary=normalized_summary,
            user_turns=tuple(turn for key, turn in turns.items() if key in retained_ids),
        )
        self._conversation_memory = next_snapshot

    def reset_conversation_memory(self) -> None:
        self._conversation_memory = ConversationMemorySnapshot()
        self.previous_response = None
        self.pending_action = None

    def commit_response_state(
        self,
        *,
        response: AnswerResponse,
        response_kind: str,
        pending_action: PendingAction | None,
    ) -> None:
        self.pending_action = pending_action.model_copy(deep=True) if pending_action is not None else None
        if response_kind == "answer":
            self.previous_response = response.model_copy(deep=True)

    def set_session_metadata(self, session_metadata: SessionMetadata | None) -> None:
        self.session_metadata = parse_session_metadata(session_metadata)

    def snapshot_session_metadata(self) -> SessionMetadata:
        return parse_session_metadata(self.session_metadata)

    def cleanup_upload_retriever(self) -> None:
        handle = self.upload_retriever_handle
        if handle is None:
            return

        try:
            handle.cleanup()
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "upload_retriever_cleanup_failed",
                collection=handle.collection_name,
                error=exc,
            )
        finally:
            self.upload_retriever_handle = None

    def close(self) -> None:
        self.cleanup_upload_retriever()
        self.upload_file_path = None
        self.reset_conversation_memory()
        self.session_metadata = parse_session_metadata(None)


__all__ = ["ConversationMemorySnapshot", "SessionContext"]
