from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from src.core.latency import StageName


PublishProgressEvent = Callable[[str, dict[str, Any]], None]


class ProgressEmitter:
    def __init__(
        self,
        *,
        publish: PublishProgressEvent,
        request_id: str,
        session_id: str,
        heartbeat_interval_seconds: float = 1.5,
    ) -> None:
        self._publish = publish
        self._request_id = str(request_id)
        self._session_id = str(session_id)
        self._heartbeat_interval_seconds = max(0.25, float(heartbeat_interval_seconds))
        self._lock = threading.Lock()
        self._current_stage: StageName | None = None
        self._current_attempt = 1
        self._current_stage_started = 0.0
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None

    def emit_request_started(self) -> None:
        self._publish(
            "request_started",
            {
                "request_id": self._request_id,
                "session_id": self._session_id,
            },
        )

    def emit_stage_started(self, *, stage: StageName, attempt: int = 1) -> None:
        started = time.monotonic()
        with self._lock:
            self._stop_heartbeat_locked()
            self._current_stage = stage
            self._current_attempt = max(1, int(attempt))
            self._current_stage_started = started
            if stage == "synthesis":
                self._start_heartbeat_locked(stage=stage, attempt=self._current_attempt)
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
        with self._lock:
            started = self._current_stage_started
            if self._current_stage == stage:
                self._stop_heartbeat_locked()
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

    def emit_error(self, *, message: str, stage: StageName | None = None) -> None:
        with self._lock:
            active_stage = stage or self._current_stage
            self._stop_heartbeat_locked()
            self._current_stage = None
            self._current_stage_started = 0.0

        payload = {"message": str(message)}
        if active_stage:
            payload["stage"] = active_stage
        self._publish("error", payload)

    def emit_final_response(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._stop_heartbeat_locked()
            self._current_stage = None
            self._current_stage_started = 0.0
        self._publish("final_response", dict(payload))

    def emit_done(self) -> None:
        with self._lock:
            self._stop_heartbeat_locked()
            self._current_stage = None
            self._current_stage_started = 0.0
        self._publish("done", {})

    def _start_heartbeat_locked(self, *, stage: StageName, attempt: int) -> None:
        stop_event = threading.Event()
        started = self._current_stage_started

        def heartbeat_loop() -> None:
            while not stop_event.wait(self._heartbeat_interval_seconds):
                elapsed_ms = max(0, int(round((time.monotonic() - started) * 1000)))
                self._publish(
                    "heartbeat",
                    {
                        "stage": stage,
                        "attempt": attempt,
                        "elapsed_ms": elapsed_ms,
                    },
                )

        thread = threading.Thread(
            target=heartbeat_loop,
            name=f"progress-heartbeat-{self._request_id}",
            daemon=True,
        )
        self._heartbeat_stop = stop_event
        self._heartbeat_thread = thread
        thread.start()

    def _stop_heartbeat_locked(self) -> None:
        stop_event = self._heartbeat_stop
        thread = self._heartbeat_thread
        self._heartbeat_stop = None
        self._heartbeat_thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.1)

