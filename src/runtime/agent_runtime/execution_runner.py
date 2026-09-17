from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
import time
import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.core.contracts.boundary.graph import build_graph_state_input, normalize_graph_update
from src.core.latency import elapsed_ms
from src.infra.settings import AppSettings
from src.infra.tools.local_rag import build_temp_retriever
from src.runtime.agent_runtime.session_context import SessionContext
from src.core.uploads import UploadContext


class GraphInvocationError(RuntimeError):
    def __init__(self, *, graph_total_ms: int, cause: Exception):
        super().__init__(str(cause))
        self.graph_total_ms = graph_total_ms
        self.cause = cause


@dataclass(slots=True)
class _UploadedRetrieverBuildResult:
    handle: Any
    build_ms: int


class PendingUploadedRetriever:
    def __init__(
        self,
        *,
        future: Future[_UploadedRetrieverBuildResult],
        executor: ThreadPoolExecutor,
        session: SessionContext,
        upload_file_path: str,
        upload_content_hash: str,
    ) -> None:
        self._future = future
        self._executor = executor
        self._session = session
        self._upload_file_path = upload_file_path
        self._upload_content_hash = upload_content_hash
        self._lock = Lock()
        self._result: _UploadedRetrieverBuildResult | None = None
        self._executor_closed = False

    def _shutdown_executor(self) -> None:
        if self._executor_closed:
            return
        self._executor_closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def cancel(self) -> None:
        with self._lock:
            already_resolved = self._result is not None
        if not already_resolved:
            def cleanup_completed_handle(future: Future[_UploadedRetrieverBuildResult]) -> None:
                try:
                    result = future.result()
                except Exception:
                    return
                cleanup = getattr(result.handle, "cleanup", None)
                if callable(cleanup):
                    cleanup()

            if not self._future.cancel():
                self._future.add_done_callback(cleanup_completed_handle)
        self._shutdown_executor()

    def resolve(self) -> _UploadedRetrieverBuildResult:
        with self._lock:
            if self._result is not None:
                return self._result

        result = self._future.result()
        with self._lock:
            if self._result is None:
                self._result = result
                self._session.upload_retriever_handle = result.handle
                self._session.upload_file_path = self._upload_file_path
                self._session.upload_content_hash = self._upload_content_hash
                self._shutdown_executor()
            return self._result

    @property
    def build_ms(self) -> int | None:
        with self._lock:
            return self._result.build_ms if self._result is not None else None

    @property
    def vectorstore(self) -> Any:
        return getattr(self.resolve().handle.retriever, "vectorstore", None)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.resolve().handle.retriever.invoke(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.resolve().handle.retriever, name)


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
        self._pending_upload_retriever: PendingUploadedRetriever | None = None

    def _start_upload_retriever_build(self, upload_file_path: str, upload_content_hash: str) -> PendingUploadedRetriever:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="documate-upload-build")

        def build() -> _UploadedRetrieverBuildResult:
            build_started = time.perf_counter()
            handle = self._build_temp_retriever(
                upload_file_path,
                api_key=self.settings.openai_api_key,
            )
            return _UploadedRetrieverBuildResult(
                handle=handle,
                build_ms=elapsed_ms(build_started, time.perf_counter()),
            )

        future = executor.submit(build)
        return PendingUploadedRetriever(
            future=future,
            executor=executor,
            session=self.session,
            upload_file_path=upload_file_path,
            upload_content_hash=upload_content_hash,
        )

    def finalize_pending_upload_retriever(self, *, wait: bool = True) -> int | None:
        pending = self._pending_upload_retriever
        self._pending_upload_retriever = None
        if pending is None:
            return None
        if not wait and pending.build_ms is None:
            pending.cancel()
            return None
        try:
            return pending.resolve().build_ms
        except Exception as exc:
            self.session.cleanup_upload_retriever()
            self.session.upload_file_path = None
            raise RuntimeError(
                f"UPLOAD_RETRIEVER_BUILD_FAILED: upload retriever build failed ({exc})"
            ) from exc

    def cancel_pending_upload_retriever(self) -> None:
        pending = self._pending_upload_retriever
        self._pending_upload_retriever = None
        if pending is not None:
            pending.cancel()

    def prepare_graph_state(
        self,
        user_input: str,
        upload_file_path: str | None,
        progress_emitter: Any | None = None,
        *, uploads: UploadContext | None = None,
    ) -> tuple[dict[str, Any], int | None]:
        self._pending_upload_retriever = None
        conversation = self.session.snapshot_conversation_memory()
        session_metadata = self.session.snapshot_session_metadata()
        current_turn_id = f"user:{uuid4().hex}"

        def build_state(retriever: Any | None = None) -> dict[str, Any]:
            return build_graph_state_input(
                session_id=self.session.session_id,
                user_input=user_input,
                current_turn_id=current_turn_id,
                user_turns=conversation.user_turns,
                messages=list(conversation.messages),
                retriever=retriever,
                upload_files=tuple(record.public_info() for record in self.session.upload_records),
                progress_emitter=progress_emitter,
                memory_summary=conversation.memory_summary,
                session_metadata=session_metadata,
                previous_response=(self.session.previous_response.model_copy(deep=True) if self.session.previous_response is not None else None),
                pending_action=(self.session.pending_action.model_copy(deep=True) if self.session.pending_action is not None else None),
            )

        upload_retriever_build_ms: int | None = None

        if uploads is not None:
            if uploads.epoch != self.session.upload_epoch or uploads.revision != self.session.upload_revision:
                raise ValueError("UPLOAD_REVISION_CONFLICT: attachment set changed")
            # A legacy retriever is outside the versioned manifest's file set.
            handle = self.session.upload_retriever_handle if self.session.upload_records else None
            return normalize_graph_update(build_state(handle.retriever if handle is not None else None)), None

        if self.session.upload_records:
            # A direct legacy call replaces the managed set, including its
            # manifest and owned originals. Retire its context before execution.
            self.session.release_upload_resources()
            self.session.upload_revision += 1
        state = build_state()

        if upload_file_path is not None:
            upload_content_hash = hashlib.sha256(Path(upload_file_path).read_bytes()).hexdigest()
            if (
                self.session.upload_file_path != upload_file_path
                or self.session.upload_content_hash != upload_content_hash
                or self.session.upload_retriever_handle is None
            ):
                self.session.cleanup_upload_retriever()
                pending_retriever = self._start_upload_retriever_build(upload_file_path, upload_content_hash)
                self._pending_upload_retriever = pending_retriever
                state = build_state(pending_retriever)
                return normalize_graph_update(state), upload_retriever_build_ms

            handle = self.session.upload_retriever_handle
            if handle is not None:
                state = build_state(handle.retriever)
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
