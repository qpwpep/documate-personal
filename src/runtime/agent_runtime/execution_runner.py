from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from src.core.contracts.boundary.graph import build_graph_state_input, normalize_graph_update
from src.core.latency import elapsed_ms
from src.runtime.agent_runtime.session_context import SessionContext


class GraphInvocationError(RuntimeError):
    def __init__(self, *, graph_total_ms: int, cause: Exception, last_state: dict[str, Any] | None):
        super().__init__(str(cause))
        self.graph_total_ms = graph_total_ms
        self.cause = cause
        self.last_state = last_state


class ExecutionRunner:
    def __init__(self, *, graph: Any, session: SessionContext) -> None:
        self.graph = graph
        self.session = session

    def prepare_graph_state(
        self,
        user_input: str,
        *,
        progress_emitter: Any | None = None,
    ) -> dict[str, Any]:
        """Read committed session state while the application holds its request lock."""
        conversation = self.session.snapshot_conversation_memory()
        handle = self.session.upload_retriever_handle
        state = build_graph_state_input(
            session_id=self.session.session_id,
            user_input=user_input,
            current_turn_id=f"user:{uuid4().hex}",
            user_turns=conversation.user_turns,
            messages=list(conversation.messages),
            retriever=handle.retriever if handle is not None else None,
            upload_files=tuple(record.public_info() for record in self.session.upload_records),
            progress_emitter=progress_emitter,
            memory_summary=conversation.memory_summary,
            session_metadata=self.session.snapshot_session_metadata(),
            previous_response=(self.session.previous_response.model_copy(deep=True)
                               if self.session.previous_response is not None else None),
            pending_action=(self.session.pending_action.model_copy(deep=True)
                            if self.session.pending_action is not None else None),
        )
        return normalize_graph_update(state)

    def invoke_graph(self, state: dict[str, Any]) -> tuple[dict[str, Any], int]:
        graph_started = time.perf_counter()
        last_state: dict[str, Any] | None = None
        try:
            # Each values event is the complete committed state. Replacing the
            # snapshot preserves completed decisions without appending them again.
            for snapshot in self.graph.stream(state, stream_mode="values"):
                last_state = snapshot
            if last_state is None:
                raise RuntimeError("Graph execution produced no state snapshots")
        except Exception as exc:
            graph_total_ms = elapsed_ms(graph_started, time.perf_counter())
            raise GraphInvocationError(
                graph_total_ms=graph_total_ms, cause=exc, last_state=last_state,
            ) from exc
        graph_total_ms = elapsed_ms(graph_started, time.perf_counter())
        return last_state, graph_total_ms
