from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from src.core.latency import StageName


PublishProgressEvent = Callable[[str, dict[str, Any]], None]


class ProgressEmitter:
    """Publish execution progress; the HTTP stream owns connection keepalives."""

    def __init__(
        self,
        *,
        publish: PublishProgressEvent,
        request_id: str,
        session_id: str,
    ) -> None:
        self._publish = publish
        self._request_id = str(request_id)
        self._session_id = str(session_id)
        self._current_stage: StageName | None = None
        self._current_stage_started = 0.0

    @property
    def request_id(self) -> str:
        return self._request_id

    def emit_request_started(self) -> None:
        self._publish(
            "request_started",
            {
                "request_id": self._request_id,
                "session_id": self._session_id,
            },
        )

    def emit_stage_started(self, *, stage: StageName, attempt: int = 1) -> None:
        self._current_stage = stage
        self._current_stage_started = time.monotonic()
        self._publish(
            "stage_started",
            {
                "stage": stage,
                "attempt": max(1, int(attempt)),
            },
        )

    def emit_stage_completed(
        self,
        *,
        stage: StageName,
        attempt: int = 1,
        latency_ms: int | None = None,
        status: str | None = None,
    ) -> None:
        started = self._current_stage_started
        if self._current_stage == stage:
            self._current_stage = None
            self._current_stage_started = 0.0
        computed_latency_ms = latency_ms
        if computed_latency_ms is None and started > 0:
            computed_latency_ms = max(0, int(round((time.monotonic() - started) * 1000)))

        payload = {
            "stage": stage,
            "attempt": max(1, int(attempt)),
        }
        if computed_latency_ms is not None:
            payload["latency_ms"] = max(0, int(computed_latency_ms))
        if status:
            payload["status"] = str(status)
        self._publish("stage_completed", payload)

    def emit_progress_snapshot(self, *, stage: StageName, summary: str, **data: Any) -> None:
        payload = {
            "stage": stage,
            "summary": str(summary or "").strip(),
        }
        for key, value in data.items():
            if value is not None:
                payload[str(key)] = value
        self._publish("progress_snapshot", payload)

    def emit_final_response(self, payload: dict[str, Any]) -> None:
        self._current_stage = None
        self._current_stage_started = 0.0
        self._publish("final_response", dict(payload))

    def emit_done(self) -> None:
        self._current_stage = None
        self._current_stage_started = 0.0
        self._publish("done", {})
