"""Record logical model attempts independently of graph state transactions."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from time import perf_counter
from typing import Literal

from src.core.contracts.debug import build_llm_call_metadata
from src.core.contracts.usage import LLMCallPath, LLMCallRecord, LLMCallStage


class LLMUsageRecorder:
    def __init__(self) -> None:
        self._calls: list[LLMCallRecord] = []
        self._lock = Lock()

    def start(self, call: LLMCallRecord) -> int:
        with self._lock:
            index = len(self._calls)
            self._calls.append(call)
            return index

    def complete(self, index: int, message: object) -> None:
        with self._lock:
            call = self._calls[index]
            observed = build_llm_call_metadata(
                stage=call.stage, attempt=call.attempt, path=call.path, message=message,
                model_name=call.model_name, endpoint=call.endpoint,
            )
            self._calls[index] = observed.model_copy(update={"status": call.status})

    def finish(self, index: int, *, status: Literal["completed", "failed"], latency_ms: int,
               failure_type: str | None) -> None:
        with self._lock:
            self._calls[index] = self._calls[index].model_copy(update={
                "status": status, "latency_ms": latency_ms, "failure_type": failure_type,
            })

    def snapshot(self) -> list[LLMCallRecord]:
        with self._lock:
            return [call.model_copy(deep=True) for call in self._calls]


_recorder: ContextVar[LLMUsageRecorder | None] = ContextVar("llm_usage_recorder", default=None)


@contextmanager
def capture_llm_usage() -> Iterator[LLMUsageRecorder]:
    recorder = LLMUsageRecorder()
    token = _recorder.set(recorder)
    try:
        yield recorder
    finally:
        _recorder.reset(token)


def current_llm_calls() -> list[LLMCallRecord] | None:
    recorder = _recorder.get()
    return recorder.snapshot() if recorder is not None else None


class LLMCallAttempt:
    def __init__(self, recorder: LLMUsageRecorder | None, index: int | None) -> None:
        self._recorder = recorder
        self._index = index

    def complete(self, message: object) -> None:
        if self._recorder is not None and self._index is not None:
            self._recorder.complete(self._index, message)


@contextmanager
def record_llm_call(*, stage: LLMCallStage, attempt: int, path: LLMCallPath,
                    model_name: str | None = None, endpoint: str | None = None) -> Iterator[LLMCallAttempt]:
    recorder = _recorder.get()
    index = recorder.start(build_llm_call_metadata(
        stage=stage, attempt=attempt, path=path, model_name=model_name, endpoint=endpoint,
    ).model_copy(update={"status": "started"})) if recorder is not None else None
    started = perf_counter()
    failure_type = None
    try:
        yield LLMCallAttempt(recorder, index)
    except BaseException as exc:
        # Preserve both observed usage and failed/cancelled attempts without retaining exception text.
        failure_type = type(exc).__name__
        raise
    finally:
        if recorder is not None and index is not None:
            recorder.finish(index, status="failed" if failure_type else "completed",
                            latency_ms=round((perf_counter() - started) * 1000), failure_type=failure_type)
