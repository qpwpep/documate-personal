from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .summary_models import RunSummary, RunTrack


@dataclass(frozen=True)
class StoredRun:
    summary: RunSummary
    generated_at: datetime

    @property
    def run_id(self) -> str:
        return self.summary.run_id

    @property
    def metrics(self):
        return self.summary.metrics

    @property
    def track(self) -> RunTrack:
        return self.summary.track


def latest_run_pointer_name(track: RunTrack) -> str:
    return f"latest_{track}_run.txt"


def suite_label(fixtures_path: str) -> str:
    file_name = Path(fixtures_path.replace("\\", "/")).name
    if file_name == "cases.generated.jsonl":
        return "generated-suite"
    return file_name


def _parse_generated_at(summary: RunSummary) -> datetime:
    return datetime.fromisoformat(summary.generated_at_utc)


def _is_historical_summary(payload: dict[str, Any]) -> bool:
    # A retained current suite or versioned decision cannot become legacy just
    # because its version fields were removed or downgraded.
    if payload.get("release_suite") is not None:
        return False
    decision_version = payload.get("decision_contract_version")
    if decision_version is None:
        if payload.get("case_policy_snapshots") or any(
            payload.get(field) is not None for field in ("results_fingerprint", "release_decision")
        ):
            return False
    elif type(decision_version) is not int or decision_version != 1:
        return False
    return payload.get("measurement_contract_version") in (
        None, "attachment-question-scenario-v1", "llm-usage-scenario-v2",
    )


def _read_summary(path: Path) -> RunSummary | None:
    from .reporting.writer import is_current_measurement, load_run_outputs

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run summary must be a JSON object")
    if _is_historical_summary(payload):
        # Exclude known historical contracts before loading their evidence.
        # Mixed/unknown versions and damaged current runs still fail validation.
        return None
    if _is_previous_contract_summary(path, payload):
        return None
    summary, _ = load_run_outputs(path.parent)
    if not is_current_measurement(summary):
        return None
    return summary


def _is_previous_contract_summary(path: Path, payload: dict[str, object]) -> bool:
    """Exclude known immutable archives after checking their raw integrity."""
    from .decisions import raw_results_fingerprint

    decision = payload.get("decision_contract_version")
    measurement = payload.get("measurement_contract_version")
    if decision != 2 and not (decision == 3 and measurement == "llm-usage-scenario-v6"):
        return False
    known = {
        "llm-usage-scenario-v3": (
            ("shared-client-scenario-v1", "release-obligations-v6"),
            ("shared-client-scenario-v2", "release-obligations-v7"),
            ("shared-client-scenario-v2", "release-obligations-v8"),
            ("shared-client-scenario-v2", "release-obligations-v9"),
        ),
        "llm-usage-scenario-v4": (("shared-client-scenario-v2", "release-obligations-v9"),),
        "llm-usage-scenario-v5": (("shared-client-scenario-v2", "release-obligations-v9"),),
    }
    if decision == 3:
        known = {"llm-usage-scenario-v6": (("shared-client-scenario-v2", "release-obligations-v10"),)}
    audit = payload.get("audit_metrics")
    versions = (payload.get("execution_contract_version"),
                audit.get("scoring_contract_version") if isinstance(audit, dict) else None)
    if not isinstance(measurement, str) or versions not in known.get(measurement, ()):
        raise ValueError("unknown or mixed historical decision contract")
    raw = [json.loads(line) for line in (path.parent / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()
           if line.strip()]
    if any(not isinstance(result, dict) or result.get("decision_contract_version") != decision
           or result.get("run_id") != payload.get("run_id") for result in raw):
        raise ValueError("historical decision contract disagrees with retained results")
    if payload.get("results_fingerprint") != raw_results_fingerprint(raw):
        raise ValueError("run results fingerprint mismatch")
    return True


def load_history_runs(output_root: Path) -> list[StoredRun]:
    from .reporting.writer import is_completed_run

    runs: list[StoredRun] = []
    for entry in output_root.iterdir():
        if not entry.is_dir():
            continue
        if not is_completed_run(entry):
            continue
        summary_path = entry / "summary.json"
        if not summary_path.exists():
            continue
        summary = _read_summary(summary_path)
        if summary is None:
            continue
        runs.append(
            StoredRun(
                summary=summary,
                generated_at=_parse_generated_at(summary),
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
    if not track_runs:
        # A filtered historical run or stale pointer cannot substitute another
        # track's measurements for the requested history.
        raise ValueError(f"No {track} benchmark summaries were found.")
    latest = next((run for run in track_runs if run.run_id == latest_run_id), None)
    if latest is None:
        latest = _default_latest_run(track_runs, track)

    fixtures_path = latest.summary.fixtures_path
    total_cases = latest.metrics.total_cases
    comparison_contract = (
        latest.summary.execution_contract_version,
        latest.summary.measurement_contract_version,
        latest.summary.suite_fingerprint,
        latest.summary.evaluation_fingerprint,
    )
    if any(comparison_contract) and not all(comparison_contract):
        return latest, [latest]
    comparable = [
        run
        for run in track_runs
        if run.summary.decision_contract_version == latest.summary.decision_contract_version
        and run.summary.fixtures_path == fixtures_path
        and run.metrics.total_cases == total_cases
        and run.summary.execution_contract_version == latest.summary.execution_contract_version
        and run.summary.measurement_contract_version == latest.summary.measurement_contract_version
        and run.summary.suite_fingerprint == latest.summary.suite_fingerprint
        and run.summary.evaluation_fingerprint == latest.summary.evaluation_fingerprint
    ]
    comparable.sort(key=lambda item: item.generated_at)
    return latest, comparable
