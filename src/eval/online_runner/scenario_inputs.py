"""Fixture and destination selection; request construction belongs to the app client."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.app.client import AgentRequestContext
from ..approved_uploads import ApprovedUpload, select_approved_uploads
from ..config_models import BenchmarkCase, BenchmarkLiveSlackConfig


@dataclass
class FixtureUpload:
    path: Path
    content: bytes | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def size(self) -> int:
        return len(self.content) if self.content is not None else self.path.stat().st_size

    def getbuffer(self) -> bytes:
        if self.content is None:
            self.content = self.path.read_bytes()
        return self.content

    @property
    def fingerprint(self) -> str | None:
        return hashlib.sha256(self.content).hexdigest() if self.content is not None else None


def resolve_fixture_uploads(fixtures_path: Path, case: BenchmarkCase, *,
                            verified_uploads: Mapping[str, ApprovedUpload] | None = None
                            ) -> list[FixtureUpload] | list[ApprovedUpload]:
    """Select approved identities, or resolve files for an unverified scenario."""
    if verified_uploads is not None:
        return select_approved_uploads(case.resolved_upload_fixtures, verified_uploads)

    root = (fixtures_path.parent / "uploads").resolve()
    files = []
    for name in case.resolved_upload_fixtures:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise FileNotFoundError(f"upload fixture not found within fixture uploads: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"upload fixture not found within fixture uploads: {name}")
        files.append(FixtureUpload(path))
    return files


def case_context(*, endpoint: str, session_id: str, case: BenchmarkCase,
                 timeout_seconds: int, live_slack: BenchmarkLiveSlackConfig) -> AgentRequestContext:
    destination = {"slack_channel_id": case.slack_channel_id or "",
                   "slack_user_id": case.slack_user_id or "", "slack_email": case.slack_email or ""}
    if live_slack.applies_to_case(case):
        destination = ({"slack_channel_id": live_slack.channel_id or ""}
                       if live_slack.requires_channel_destination(case) else live_slack.resolve_dm_payload())
    return AgentRequestContext(fastapi_url=endpoint, session_id=session_id,
                               include_debug=True, timeout_seconds=timeout_seconds, **destination)
