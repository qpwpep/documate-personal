from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..io import dump_jsonl
from ..result_models import CaseResult
from ..summary_models import RunSummary
from .markdown import build_markdown_report


@dataclass(frozen=True)
class ReportInputs:
    """Keep historical observations separate from current decision projections."""

    summary: RunSummary
    current_results: list[CaseResult] | None = None
    historical_results: list[dict[str, Any]] | None = None


def write_run_outputs(*, output_dir: Path, results: list[CaseResult], summary: RunSummary) -> None:
    from ..decisions import validate_run_outputs

    validate_run_outputs(summary, results)
    report = build_markdown_report(summary, results)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_dir / "raw_results.jsonl", results)
    (output_dir / "summary.json").write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
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
    """Read a current run and verify its saved evidence without calling providers."""
    from ..decisions import validate_run_outputs

    summary_path = output_dir / "summary.json"
    raw_path = output_dir / "raw_results.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_results.jsonl not found: {raw_path}")
    summary = RunSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    results = [
        CaseResult.model_validate_json(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_run_outputs(summary, results)
    return summary, results


def load_report_inputs(output_dir: Path) -> ReportInputs:
    """Read a report without upgrading historical facts to current eligibility."""
    summary_path = output_dir / "summary.json"
    raw_path = output_dir / "raw_results.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_results.jsonl not found: {raw_path}")
    summary = RunSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    if summary.decision_contract_version is not None:
        summary, results = load_run_outputs(output_dir)
        return ReportInputs(summary=summary, current_results=results)

    historical_results: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("historical case results must be JSON objects")
        if payload.get("decision_contract_version") is not None:
            raise ValueError("historical run contains versioned results")
        # The evaluation model deliberately derives an ineligible current verdict
        # for legacy inputs. Use a copy only to validate their structure; its
        # derived verdict must never replace the historical record used by reports.
        parsed = CaseResult.model_validate(deepcopy(payload))
        if parsed.run_id != summary.run_id:
            raise ValueError("historical case result belongs to a different run")
        case_ids.add(parsed.case_id)
        historical_results.append(payload)
    if len(historical_results) != summary.metrics.total_cases:
        raise ValueError("historical result count does not match the saved summary")
    duplicates = len(historical_results) - len(case_ids)
    if duplicates != summary.metrics.duplicate_result_cases:
        raise ValueError("historical duplicate count does not match the saved summary")
    return ReportInputs(summary=summary, historical_results=historical_results)


__all__ = [
    "ReportInputs",
    "load_report_inputs",
    "load_run_outputs",
    "write_run_outputs",
]
