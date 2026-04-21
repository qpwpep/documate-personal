from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .summary_models import RunSummary, RunTrack


@dataclass(frozen=True)
class StoredRun:
    summary: RunSummary
    generated_at: datetime
    track_explicit: bool = False

    @property
    def run_id(self) -> str:
        return self.summary.run_id

    @property
    def metrics(self):
        return self.summary.metrics

    @property
    def track(self) -> RunTrack:
        if self.track_explicit:
            return self.summary.track
        if self.summary.requested_limit is not None:
            return "smoke"
        return "release"


def latest_run_pointer_name(track: RunTrack) -> str:
    return f"latest_{track}_run.txt"


def suite_label(fixtures_path: str) -> str:
    file_name = Path(fixtures_path.replace("\\", "/")).name
    if file_name == "cases.generated.jsonl":
        return "generated-suite"
    return file_name


def _parse_generated_at(summary: RunSummary) -> datetime:
    return datetime.fromisoformat(summary.generated_at_utc)


def _read_summary(path: Path) -> tuple[RunSummary, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunSummary(**payload), "track" in payload


def load_history_runs(output_root: Path) -> list[StoredRun]:
    runs: list[StoredRun] = []
    for entry in output_root.iterdir():
        if not entry.is_dir():
            continue
        summary_path = entry / "summary.json"
        if not summary_path.exists():
            continue
        summary, track_explicit = _read_summary(summary_path)
        runs.append(
            StoredRun(
                summary=summary,
                generated_at=_parse_generated_at(summary),
                track_explicit=track_explicit,
            )
        )
    runs.sort(key=lambda item: item.generated_at)
    return runs


def load_latest_run_id(output_root: Path, *, track: RunTrack) -> str | None:
    latest_path = output_root / latest_run_pointer_name(track)
    if not latest_path.exists():
        return None
    latest_run_id = latest_path.read_text(encoding="utf-8").strip()
    return latest_run_id or None


def _default_latest_run(runs: list[StoredRun], track: RunTrack) -> StoredRun:
    if track == "release":
        max_total_cases = max(run.metrics.total_cases for run in runs)
        release_like_runs = [run for run in runs if run.metrics.total_cases == max_total_cases]
        return release_like_runs[-1]
    return runs[-1]


def select_comparable_runs(
    runs: list[StoredRun],
    *,
    track: RunTrack,
    latest_run_id: str | None = None,
) -> tuple[StoredRun, list[StoredRun]]:
    if not runs:
        raise ValueError("No benchmark summaries were found.")

    track_runs = [run for run in runs if run.track == track]
    if track_runs:
        latest = next((run for run in track_runs if run.run_id == latest_run_id), None)
        if latest is None:
            latest = _default_latest_run(track_runs, track)
    else:
        if track == "smoke":
            raise ValueError("No smoke benchmark summaries were found.")
        latest = next((run for run in runs if run.run_id == latest_run_id), None)
        if latest is None:
            latest = _default_latest_run(runs, track)

    fixtures_path = latest.summary.fixtures_path
    total_cases = latest.metrics.total_cases
    comparable = [
        run
        for run in runs
        if run.track == latest.track
        and run.summary.fixtures_path == fixtures_path
        and run.metrics.total_cases == total_cases
    ]
    comparable.sort(key=lambda item: item.generated_at)
    return latest, comparable
