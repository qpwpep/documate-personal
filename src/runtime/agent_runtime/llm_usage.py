"""Record logical model attempts independently of graph state transactions."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any

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

    def complete(self, index: int, message: Any) -> None:
        with self._lock:
            call = self._calls[index]
            self._calls[index] = build_llm_call_metadata(
                stage=call.stage, attempt=call.attempt, path=call.path, message=message,
            )

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

    def complete(self, message: Any) -> None:
        if self._recorder is not None and self._index is not None:
            self._recorder.complete(self._index, message)


@contextmanager
def record_llm_call(*, stage: LLMCallStage, attempt: int, path: LLMCallPath) -> Iterator[LLMCallAttempt]:
    recorder = _recorder.get()
    index = recorder.start(build_llm_call_metadata(
        stage=stage, attempt=attempt, path=path, message=None,
    )) if recorder is not None else None
    # Exceptions leave the initial, explicitly unobserved usage record intact.
    yield LLMCallAttempt(recorder, index)
