from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ..logging_utils import log_event
from ..runtime_paths import get_service_state_path
from ._bootstrap import logger


STATE_SCHEMA_VERSION = 2


def _maybe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class ServiceState:
    schema_version: int = STATE_SCHEMA_VERSION
    fastapi_pid: int | None = None
    fastapi_create_time: float | None = None
    streamlit_pid: int | None = None
    streamlit_create_time: float | None = None
    fastapi_log: str | None = None
    streamlit_log: str | None = None
    fastapi_port: int | None = None
    streamlit_port: int | None = None
    started_at_unix: int | None = None
    platform: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ServiceState":
        return cls(
            schema_version=_maybe_int(payload.get("schema_version")) or STATE_SCHEMA_VERSION,
            fastapi_pid=_maybe_int(payload.get("fastapi_pid")),
            fastapi_create_time=_maybe_float(payload.get("fastapi_create_time")),
            streamlit_pid=_maybe_int(payload.get("streamlit_pid")),
            streamlit_create_time=_maybe_float(payload.get("streamlit_create_time")),
            fastapi_log=str(payload.get("fastapi_log") or "").strip() or None,
            streamlit_log=str(payload.get("streamlit_log") or "").strip() or None,
            fastapi_port=_maybe_int(payload.get("fastapi_port")),
            streamlit_port=_maybe_int(payload.get("streamlit_port")),
            started_at_unix=_maybe_int(payload.get("started_at_unix")),
            platform=str(payload.get("platform") or "").strip() or None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fastapi_pid": self.fastapi_pid,
            "fastapi_create_time": self.fastapi_create_time,
            "streamlit_pid": self.streamlit_pid,
            "streamlit_create_time": self.streamlit_create_time,
            "fastapi_log": self.fastapi_log,
            "streamlit_log": self.streamlit_log,
            "fastapi_port": self.fastapi_port,
            "streamlit_port": self.streamlit_port,
            "started_at_unix": self.started_at_unix,
            "platform": self.platform,
        }


def load_service_state() -> ServiceState | None:
    state_path = get_service_state_path()
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log_event(logger, logging.WARNING, "service_state_load_failed", state_path=state_path, error=exc)
        return None
    if not isinstance(payload, dict):
        log_event(logger, logging.WARNING, "service_state_load_failed", state_path=state_path, error="invalid_state_payload")
        return None
    return ServiceState.from_dict(payload)


def save_service_state(state: ServiceState) -> None:
    state_path = get_service_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def remove_service_state() -> None:
    state_path = get_service_state_path()
    if state_path.exists():
        state_path.unlink()
