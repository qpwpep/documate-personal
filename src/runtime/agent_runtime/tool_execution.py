"""Record before tool entry, independently of graph-node update transactions."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any
from uuid import uuid4

from src.core.contracts.tool_execution import ToolExecutionEvent, ToolExecutionEvidence


class ToolExecutionRecorder:
    def __init__(self, request_id: str | None = None):
        self.request_id = request_id or uuid4().hex
        self._events: list[ToolExecutionEvent] = []
        self._open: set[str] = set()
        self._lock = Lock()

    def record(self, tool_name: str, phase: str, *, invocation_id: str | None = None,
               reason_code: str | None = None, origin_invocation_id: str | None = None) -> str:
        invocation_id = invocation_id or uuid4().hex
        with self._lock:
            event = ToolExecutionEvent(sequence=len(self._events) + 1, invocation_id=invocation_id,
                                       tool_name=tool_name, phase=phase, reason_code=reason_code,
                                       origin_invocation_id=origin_invocation_id)
            self._events.append(event)
            if phase == "started":
                self._open.add(invocation_id)
            elif phase in {"succeeded", "failed"}:
                self._open.discard(invocation_id)
        return invocation_id

    def snapshot(self) -> ToolExecutionEvidence:
        with self._lock:
            missing_origin = any(event.phase == "reused" and not event.origin_invocation_id for event in self._events)
            return ToolExecutionEvidence(schema_version=1, request_id=self.request_id,
                                         status="incomplete" if self._open or missing_origin else "complete",
                                         events=list(self._events))


_recorder: ContextVar[ToolExecutionRecorder | None] = ContextVar("tool_execution_recorder", default=None)


@contextmanager
def capture_tool_execution(request_id: str | None = None) -> Iterator[ToolExecutionRecorder]:
    recorder = ToolExecutionRecorder(request_id)
    token = _recorder.set(recorder)
    try:
        yield recorder
    finally:
        _recorder.reset(token)


def current_execution_evidence() -> ToolExecutionEvidence | None:
    recorder = _recorder.get()
    return recorder.snapshot() if recorder is not None else None


def record_nonexecution(tool_name: str, *, phase: str = "blocked", reason_code: str,
                        origin_invocation_id: str | None = None) -> str | None:
    recorder = _recorder.get()
    if recorder is not None:
        return recorder.record(tool_name, phase, reason_code=reason_code, origin_invocation_id=origin_invocation_id)
    return None


def invoke_tool(tool_name: str, implementation: Callable[..., Any], *,
                execution_id: str | None = None, **kwargs: Any) -> Any:
    recorder = _recorder.get()
    if recorder is None:
        return implementation(**kwargs)
    invocation_id = recorder.record(tool_name, "started", invocation_id=execution_id)
    try:
        result = implementation(**kwargs)
    except BaseException as exc:
        recorder.record(tool_name, "failed", invocation_id=invocation_id,
                        reason_code=str(getattr(exc, "code", None) or type(exc).__name__))
        raise
    payload = result if isinstance(result, dict) else result.model_dump() if hasattr(result, "model_dump") else {}
    diagnostic = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else payload
    status = diagnostic.get("status")
    failed = status in {"error", "unavailable", "unknown", "not_sent"}
    recorder.record(tool_name, "failed" if failed else "succeeded", invocation_id=invocation_id,
                    reason_code=diagnostic.get("error_code") or (str(status) if failed else None))
    return result
