from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schemas import BenchmarkCase


@dataclass(slots=True)
class RequestContext:
    session_id: str
    created_at: str
    request_payload: dict[str, Any]
    runtime_errors: list[str] = field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_upload_path(
    *,
    case: BenchmarkCase,
    fixtures_path: Path,
    session_id: str,
) -> str | None:
    if not case.upload_fixture:
        return None

    source = (fixtures_path.parent / "uploads" / case.upload_fixture).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"upload fixture not found: {source}")

    session_dir = (Path("uploads") / session_id).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    target = session_dir / source.name
    shutil.copy2(source, target)
    return target.as_posix()


def build_request_context(
    *,
    fixtures_path: Path,
    case: BenchmarkCase,
) -> RequestContext:
    session_id = str(uuid.uuid4())
    request_payload: dict[str, Any] = {
        "query": case.query,
        "session_id": session_id,
        "include_debug": True,
    }
    if case.slack_channel_id:
        request_payload["slack_channel_id"] = case.slack_channel_id
    if case.slack_user_id:
        request_payload["slack_user_id"] = case.slack_user_id
    if case.slack_email:
        request_payload["slack_email"] = case.slack_email

    runtime_errors: list[str] = []
    try:
        upload_path = _build_upload_path(case=case, fixtures_path=fixtures_path, session_id=session_id)
        if upload_path:
            request_payload["upload_file_path"] = upload_path
    except Exception as exc:
        runtime_errors.append(str(exc))

    return RequestContext(
        session_id=session_id,
        created_at=_utc_now_iso(),
        request_payload=request_payload,
        runtime_errors=runtime_errors,
    )


def cleanup_session_upload_dir(session_id: str) -> None:
    session_dir = Path("uploads") / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)


__all__ = [
    "RequestContext",
    "build_request_context",
    "cleanup_session_upload_dir",
]
