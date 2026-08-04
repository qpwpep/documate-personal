from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.core.contracts import SessionMetadata
from src.core.contracts.boundary.runtime import parse_session_metadata
from src.infra.logging_utils import log_event
from src.infra.tools.local_rag import UploadedRetrieverHandle


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConversationMemorySnapshot:
    messages: tuple[Any, ...] = ()
    memory_summary: str | None = None


def _clone_messages(messages: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(copy.deepcopy(message) for message in messages)


class SessionContext:
    def __init__(self) -> None:
        self._conversation_memory = ConversationMemorySnapshot()
        self.session_metadata: SessionMetadata = parse_session_metadata(None)
        self.upload_retriever_handle: UploadedRetrieverHandle | None = None
        self.upload_file_path: str | None = None

    @property
    def messages(self) -> list[Any]:
        return list(_clone_messages(self._conversation_memory.messages))

    @messages.setter
    def messages(self, value: Iterable[Any]) -> None:
        self.commit_conversation_memory(
            messages=value,
            memory_summary=self._conversation_memory.memory_summary,
        )

    @property
    def memory_summary(self) -> str | None:
        return self._conversation_memory.memory_summary

    @memory_summary.setter
    def memory_summary(self, value: str | None) -> None:
        self.commit_conversation_memory(
            messages=self._conversation_memory.messages,
            memory_summary=value,
        )

    def snapshot_conversation_memory(self) -> ConversationMemorySnapshot:
        return ConversationMemorySnapshot(
            messages=_clone_messages(self._conversation_memory.messages),
            memory_summary=self._conversation_memory.memory_summary,
        )

    def commit_conversation_memory(
        self,
        *,
        messages: Iterable[Any],
        memory_summary: str | None,
    ) -> None:
        normalized_summary = str(memory_summary or "").strip() or None
        next_snapshot = ConversationMemorySnapshot(
            messages=_clone_messages(messages),
            memory_summary=normalized_summary,
        )
        self._conversation_memory = next_snapshot

    def reset_conversation_memory(self) -> None:
        self._conversation_memory = ConversationMemorySnapshot()

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
