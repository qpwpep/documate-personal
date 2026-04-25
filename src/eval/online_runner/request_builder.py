from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.infra.runtime_paths import get_upload_session_dir
from ..config_models import BenchmarkCase, BenchmarkLiveSlackConfig


@dataclass(slots=True)
class RequestContext:
    session_id: str
    created_at: str
    request_payload: dict[str, Any]
    runtime_errors: list[str] = field(default_factory=list)
    slack_delivery_required: bool = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_upload_paths(
    *,
    upload_fixtures: list[str],
    fixtures_path: Path,
    session_id: str,
) -> list[str]:
    if not upload_fixtures:
        return []

    session_dir = get_upload_session_dir(session_id).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    upload_paths: list[str] = []
    for fixture in upload_fixtures:
        source = (fixtures_path.parent / "uploads" / fixture).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"upload fixture not found: {source}")
        target = session_dir / source.name
        shutil.copy2(source, target)
        upload_paths.append(target.as_posix())
    return upload_paths


def build_request_context(
    *,
    fixtures_path: Path,
    case: BenchmarkCase,
    live_slack: BenchmarkLiveSlackConfig | None = None,
    session_id: str | None = None,
    clear_uploads: bool = False,
) -> RequestContext:
    session_id = session_id or str(uuid.uuid4())
    request_payload: dict[str, Any] = {
        "query": case.query,
        "session_id": session_id,
        "include_debug": True,
        "planner_mode": case.planner_mode,
    }
    if case.faults:
        request_payload["eval_faults"] = dict(case.faults)
    if case.reset_slack_destination:
        request_payload["reset_slack_destination"] = True
    resolved_live_slack = live_slack or BenchmarkLiveSlackConfig()
    slack_delivery_required = False
    if resolved_live_slack.applies_to_case(case):
        slack_delivery_required = True
        if resolved_live_slack.requires_channel_destination(case):
            if resolved_live_slack.channel_id:
                request_payload["slack_channel_id"] = resolved_live_slack.channel_id
        else:
            request_payload.update(resolved_live_slack.resolve_dm_payload())
    else:
        if case.slack_channel_id:
            request_payload["slack_channel_id"] = case.slack_channel_id
        if case.slack_user_id:
            request_payload["slack_user_id"] = case.slack_user_id
        if case.slack_email:
            request_payload["slack_email"] = case.slack_email

    runtime_errors: list[str] = []
    try:
        upload_path_fault = str(case.faults.get("upload_path") or "").strip().lower()
        if upload_path_fault == "invalid":
            request_payload["upload_file_path"] = (
                fixtures_path.parent / "uploads" / "sample_data_analysis.py"
            ).resolve().as_posix()
        elif upload_path_fault == "missing":
            request_payload["upload_file_path"] = (
                get_upload_session_dir(session_id).resolve() / "missing_upload.py"
            ).as_posix()
        elif clear_uploads:
            request_payload["upload_file_paths"] = []
        else:
            upload_paths = _build_upload_paths(
                upload_fixtures=case.upload_fixtures,
                fixtures_path=fixtures_path,
                session_id=session_id,
            )
            if upload_paths:
                request_payload["upload_file_paths"] = upload_paths
                if len(upload_paths) == 1:
                    request_payload["upload_file_path"] = upload_paths[0]
    except Exception as exc:
        runtime_errors.append(str(exc))

    return RequestContext(
        session_id=session_id,
        created_at=_utc_now_iso(),
        request_payload=request_payload,
        runtime_errors=runtime_errors,
        slack_delivery_required=slack_delivery_required,
    )


def cleanup_session_upload_dir(session_id: str) -> None:
    session_dir = get_upload_session_dir(session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)


__all__ = [
    "RequestContext",
    "build_request_context",
    "cleanup_session_upload_dir",
]
