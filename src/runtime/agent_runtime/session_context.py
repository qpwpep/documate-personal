from __future__ import annotations

import logging
from typing import Any

from src.core.contracts import SessionMetadata
from src.core.contracts.boundary.runtime import parse_session_metadata
from src.infra.logging_utils import log_event
from src.infra.tools.local_rag import UploadedRetrieverHandle


logger = logging.getLogger(__name__)


class SessionContext:
    def __init__(self) -> None:
        self.messages: list[Any] = []
        self.session_metadata: SessionMetadata = parse_session_metadata(None)
        self.upload_retriever_handle: UploadedRetrieverHandle | None = None
        self.upload_file_path: str | None = None

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
        self.messages = []
        self.session_metadata = parse_session_metadata(None)
