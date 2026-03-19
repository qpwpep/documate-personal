from __future__ import annotations

import time
from typing import Any

from ..contracts.boundary.graph import build_graph_state_input, normalize_graph_update
from ..latency import elapsed_ms
from ..settings import AppSettings
from ..tools.local_rag import build_temp_retriever
from .session_context import SessionContext


class GraphInvocationError(RuntimeError):
    def __init__(self, *, graph_total_ms: int, cause: Exception):
        super().__init__(str(cause))
        self.graph_total_ms = graph_total_ms
        self.cause = cause


class ExecutionRunner:
    def __init__(
        self,
        *,
        settings: AppSettings,
        graph: Any,
        session: SessionContext,
        build_temp_retriever_fn: Any = build_temp_retriever,
    ) -> None:
        self.settings = settings
        self.graph = graph
        self.session = session
        self._build_temp_retriever = build_temp_retriever_fn

    def prepare_graph_state(
        self,
        user_input: str,
        upload_file_path: str | None,
    ) -> tuple[dict[str, Any], int | None]:
        state = build_graph_state_input(
            user_input=user_input,
            messages=self.session.messages,
            session_metadata=self.session.snapshot_session_metadata(),
        )
        upload_retriever_build_ms: int | None = None

        if upload_file_path is not None:
            if (
                self.session.upload_file_path != upload_file_path
                or self.session.upload_retriever_handle is None
            ):
                self.session.cleanup_upload_retriever()
                build_started = time.perf_counter()
                self.session.upload_retriever_handle = self._build_temp_retriever(
                    upload_file_path,
                    api_key=self.settings.openai_api_key,
                )
                upload_retriever_build_ms = elapsed_ms(build_started, time.perf_counter())
                self.session.upload_file_path = upload_file_path

            handle = self.session.upload_retriever_handle
            if handle is not None:
                state = build_graph_state_input(
                    user_input=user_input,
                    messages=self.session.messages,
                    retriever=handle.retriever,
                    session_metadata=self.session.snapshot_session_metadata(),
                )
        else:
            self.session.cleanup_upload_retriever()
            self.session.upload_file_path = None

        return normalize_graph_update(state), upload_retriever_build_ms

    def invoke_graph(self, state: dict[str, Any]) -> tuple[dict[str, Any], int]:
        graph_started = time.perf_counter()
        try:
            response = self.graph.invoke(state)
        except Exception as exc:
            graph_total_ms = elapsed_ms(graph_started, time.perf_counter())
            raise GraphInvocationError(graph_total_ms=graph_total_ms, cause=exc) from exc
        graph_total_ms = elapsed_ms(graph_started, time.perf_counter())
        return response, graph_total_ms
