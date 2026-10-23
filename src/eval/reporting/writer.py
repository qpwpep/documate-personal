from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..decisions import DECISION_CONTRACT_VERSION, raw_results_fingerprint
from ..io import dump_jsonl
from ..result_models import CaseResult
from ..summary_models import RunSummary
from .markdown import build_markdown_report


@dataclass(frozen=True)
class ReportInputs:
    """A report regenerated only from validated current-contract observations."""

    summary: RunSummary
    current_results: list[CaseResult] | None = None
    historical_results: list[dict[str, Any]] | None = None
    historical_summary: dict[str, Any] | None = None


def is_current_measurement(summary: RunSummary) -> bool:
    from .summary import EXECUTION_CONTRACT_VERSION, MEASUREMENT_CONTRACT_VERSION, SCORING_CONTRACT_VERSION

    return (summary.measurement_contract_version == MEASUREMENT_CONTRACT_VERSION
            and summary.decision_contract_version == DECISION_CONTRACT_VERSION
            and summary.execution_contract_version == EXECUTION_CONTRACT_VERSION
            and (summary.audit_metrics or {}).get("scoring_contract_version") == SCORING_CONTRACT_VERSION)


def write_run_outputs(*, output_dir: Path, results: list[CaseResult], summary: RunSummary) -> None:
    from ..decisions import validate_run_outputs

    validate_run_outputs(summary, results)
    report = build_markdown_report(summary, results)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_dir / "raw_results.jsonl", results)
    summary_payload = summary.model_dump(exclude={"weights"} if summary.weights is None else set())
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    dump_jsonl(
        output_dir / "request_map.jsonl",
        [
            {
                "run_id": result.run_id,
                "case_id": result.case_id,
                "session_id": result.session_id,
                "request_id": result.request_id,
                "query": result.query[:240],
                "query_length": len(result.query),
                "query_hash": hashlib.sha256(result.query.encode("utf-8")).hexdigest(),
                "trace": result.trace,
                "created_at_utc": result.created_at_utc,
            }
            for result in results
        ],
    )


def load_run_outputs(output_dir: Path) -> tuple[RunSummary, list[CaseResult]]:
    """Verify saved evidence for the current or a recognized prior execution contract."""
    from ..decisions import validate_run_outputs

    _, raw_results, summary = _read_saved_payloads(output_dir)
    results = [CaseResult.model_validate(payload) for payload in raw_results]
    validate_run_outputs(summary, results)
    return summary, results


def _read_saved_payloads(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], RunSummary]:
    """Verify the original fingerprint before any defaults or typed projection."""
    from .summary import EXECUTION_CONTRACT_VERSION, MEASUREMENT_CONTRACT_VERSION, SCORING_CONTRACT_VERSION

    summary_path = output_dir / "summary.json"
    raw_path = output_dir / "raw_results.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_results.jsonl not found: {raw_path}")
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_payload, dict):
        raise ValueError("run summary must be a JSON object")
    if (summary_payload.get("decision_contract_version") != DECISION_CONTRACT_VERSION
            or summary_payload.get("measurement_contract_version") != MEASUREMENT_CONTRACT_VERSION):
        raise ValueError(
            "only the current decision and measurement contracts can be loaded or regenerated; "
            "historical reports remain available as their originally saved files"
        )
    audit_metrics = summary_payload.get("audit_metrics")
    execution_and_scoring = (
        summary_payload.get("execution_contract_version"),
        audit_metrics.get("scoring_contract_version") if isinstance(audit_metrics, dict) else None,
    )
    if execution_and_scoring not in (
        (EXECUTION_CONTRACT_VERSION, SCORING_CONTRACT_VERSION),
        ("shared-client-scenario-v1", "release-obligations-v6"),
        ("shared-client-scenario-v2", "release-obligations-v7"),
        ("shared-client-scenario-v2", "release-obligations-v8"),
    ):
        raise ValueError("unknown, mixed, or missing execution and scoring contracts")
    raw_results: list[dict[str, Any]] = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("case results must be JSON objects")
        raw_results.append(payload)
    if summary_payload.get("results_fingerprint") != raw_results_fingerprint(raw_results):
        raise ValueError("run results fingerprint mismatch")
    summary = RunSummary.model_validate(summary_payload)
    return summary_payload, raw_results, summary


def load_report_inputs(output_dir: Path) -> ReportInputs:
    """Use the same evidence validation for approval loading and report rendering."""
    summary, results = load_run_outputs(output_dir)
    if not is_current_measurement(summary):
        raise ValueError(
            "only the current execution and scoring contracts can be regenerated; "
            "historical reports remain available as their originally saved files"
        )
    return ReportInputs(summary=summary, current_results=results)


__all__ = [
    "ReportInputs",
    "load_report_inputs",
    "load_run_outputs",
    "is_current_measurement",
    "write_run_outputs",
]
