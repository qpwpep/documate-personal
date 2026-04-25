from __future__ import annotations

import time
from typing import Any

from src.core.contracts.boundary.graph import build_graph_state_input, normalize_graph_update
from src.core.latency import elapsed_ms
from src.infra.settings import AppSettings
from src.infra.tools.local_rag import build_temp_retriever
from src.runtime.agent_runtime.session_context import SessionContext


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
        *,
        upload_file_paths: list[str] | None = None,
        planner_mode: str = "auto",
        eval_faults: dict[str, str] | None = None,
        progress_emitter: Any | None = None,
    ) -> tuple[dict[str, Any], int | None]:
        upload_retriever_build_ms: int | None = None
        requested_upload_paths = self._normalize_requested_upload_paths(
            upload_file_path=upload_file_path,
            upload_file_paths=upload_file_paths,
        )

        if requested_upload_paths is not None:
            if not requested_upload_paths:
                self.session.cleanup_upload_retriever()
                self.session.upload_file_path = None
                self.session.upload_file_paths = ()
            elif (
                self.session.upload_file_paths != requested_upload_paths
                or self.session.upload_retriever_handle is None
            ):
                self.session.cleanup_upload_retriever()
                build_started = time.perf_counter()
                try:
                    self.session.upload_retriever_handle = self._build_temp_retriever(
                        list(requested_upload_paths),
                        api_key=self.settings.openai_api_key,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"UPLOAD_RETRIEVER_BUILD_FAILED: upload retriever build failed ({exc})"
                    ) from exc
                upload_retriever_build_ms = elapsed_ms(build_started, time.perf_counter())
                self.session.upload_file_paths = requested_upload_paths
                self.session.upload_file_path = requested_upload_paths[0]

        handle = self.session.upload_retriever_handle
        retriever = handle.retriever if handle is not None else None
        state = build_graph_state_input(
            user_input=user_input,
            messages=self.session.messages,
            retriever=retriever,
            progress_emitter=progress_emitter,
            session_metadata=self.session.snapshot_session_metadata(),
            planner_mode=planner_mode,
            eval_faults=eval_faults,
        )

        return normalize_graph_update(state), upload_retriever_build_ms

    @staticmethod
    def _normalize_requested_upload_paths(
        *,
        upload_file_path: str | None,
        upload_file_paths: list[str] | None,
    ) -> tuple[str, ...] | None:
        if upload_file_paths is None and upload_file_path is None:
            return None
        raw_paths: list[str] = []
        if upload_file_path:
            raw_paths.append(upload_file_path)
        if upload_file_paths is not None:
            raw_paths.extend(str(path) for path in upload_file_paths if str(path).strip())
        if upload_file_paths == [] and upload_file_path is None:
            return ()
        seen: set[str] = set()
        normalized_paths: list[str] = []
        for raw_path in raw_paths:
            text = str(raw_path).strip()
            if not text or text in seen:
                continue
            normalized_paths.append(text)
            seen.add(text)
        return tuple(normalized_paths)

    def invoke_graph(self, state: dict[str, Any]) -> tuple[dict[str, Any], int]:
        graph_started = time.perf_counter()
        try:
            response = self.graph.invoke(state)
        except Exception as exc:
            graph_total_ms = elapsed_ms(graph_started, time.perf_counter())
            raise GraphInvocationError(graph_total_ms=graph_total_ms, cause=exc) from exc
        graph_total_ms = elapsed_ms(graph_started, time.perf_counter())
        return response, graph_total_ms
