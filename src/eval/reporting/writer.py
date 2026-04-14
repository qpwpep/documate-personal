from __future__ import annotations

import json
from pathlib import Path

from ..schemas import CaseResult, RunSummary, dump_jsonl
from .markdown import build_markdown_report


def write_run_outputs(*, output_dir: Path, results: list[CaseResult], summary: RunSummary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_dir / "raw_results.jsonl", results)
    (output_dir / "summary.json").write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(build_markdown_report(summary, results), encoding="utf-8")
    dump_jsonl(
        output_dir / "request_map.jsonl",
        [
            {
                "run_id": result.run_id,
                "case_id": result.case_id,
                "session_id": result.session_id,
                "request_id": result.request_id,
                "trace": result.trace,
                "created_at_utc": result.created_at_utc,
            }
            for result in results
        ],
    )


__all__ = [
    "write_run_outputs",
]
