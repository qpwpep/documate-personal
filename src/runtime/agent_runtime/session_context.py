from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4
from langchain_core.messages import HumanMessage
from src.core.answer_schema import AnswerResponse
from src.core.request_contracts import UserTurnSnapshot, required_contract_turn_ids
from src.core.conversation_memory import extract_memory_text
from src.core.uploads import UploadManifest, UploadRecord, UploadSyncResponse

from src.core.contracts import SessionMetadata
from src.core.contracts.graph_state import PendingAction
from src.core.contracts.boundary.runtime import parse_session_metadata
from src.infra.logging_utils import log_event
from src.infra.tools.local_rag import UploadedRetrieverHandle
from src.infra.upload_storage import UploadStorage, remove_managed_upload_files, clear_auxiliary_upload_files


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
        self.upload_epoch = uuid4().hex
        self.upload_revision = 0
        self.upload_records: tuple[UploadRecord, ...] = ()
        self._upload_storage: UploadStorage | None = None
        self.upload_operations: OrderedDict[str, tuple[str, UploadSyncResponse]] = OrderedDict()
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

    def upload_manifest(self) -> UploadManifest:
        return UploadManifest(epoch=self.upload_epoch, revision=self.upload_revision,
                              files=[record.public_info() for record in self.upload_records])

    def bind_upload_storage(self, session_id: str) -> UploadStorage:
        storage = UploadStorage.bind(session_id)
        if self._upload_storage is not None and self._upload_storage != storage:
            raise ValueError("A session cannot acquire another session's upload storage")
        self._upload_storage = storage
        return storage

    def replace_upload_resources(
        self, session_id: str, records: Iterable[UploadRecord], handle: UploadedRetrieverHandle | None,
    ) -> None:
        """Publish a complete candidate before releasing the retired generation.

        The caller already owns the session lock. Shared immutable originals stay
        owned by the new set; candidate rollback remains the builder's responsibility.
        """
        storage = self.bind_upload_storage(session_id)
        records = tuple(records)
        if any(not storage.owns(Path(record.path)) for record in records):
            raise ValueError("Committed upload files must belong to this session's managed storage")
        old_handle, old_records = self.upload_retriever_handle, self.upload_records
        self.upload_records = records
        self.upload_retriever_handle = handle
        self.upload_file_path = None
        self.upload_content_hash = None
        self.upload_revision += 1
        if old_handle is not None and old_handle is not handle:
            self._release_upload_handle(old_handle)
        retained = {record.path for record in records}
        remove_managed_upload_files(storage, (record.path for record in old_records if record.path not in retained))
        if not records:
            clear_auxiliary_upload_files(storage)

    @staticmethod
    def _release_upload_handle(handle: UploadedRetrieverHandle) -> None:
        try:
            handle.cleanup()
        except Exception as exc:
            log_event(logger, logging.WARNING, "upload_retriever_cleanup_failed",
                      collection=handle.collection_name, error=exc)

    def cleanup_upload_retriever(self) -> None:
        handle = self.upload_retriever_handle
        if handle is None:
            return

        self.upload_retriever_handle = None
        self._release_upload_handle(handle)

    def release_upload_resources(self) -> None:
        """Release only owned originals; direct legacy input paths remain borrowed."""
        records, self.upload_records = self.upload_records, ()
        self.cleanup_upload_retriever()
        self.upload_file_path = None
        self.upload_content_hash = None
        if self._upload_storage is not None:
            remove_managed_upload_files(self._upload_storage, (record.path for record in records))
            clear_auxiliary_upload_files(self._upload_storage)

    def close(self) -> None:
        self.release_upload_resources()
        self.upload_operations.clear()
        self.upload_epoch = uuid4().hex
        self.upload_revision = 0
        self.reset_conversation_memory()
        self.session_metadata = parse_session_metadata(None)


__all__ = ["ConversationMemorySnapshot", "SessionContext"]
