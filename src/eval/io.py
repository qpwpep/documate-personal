from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config_models import BenchmarkCase, BenchmarkConfig


def load_cases_jsonl(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        record = line.strip()
        if not record:
            continue
        context = f"{path}:{line_number}"
        try:
            payload = json.loads(record)
            if isinstance(payload, dict) and "case_id" in payload:
                context += f" (case {payload['case_id']!r})"
            cases.append(BenchmarkCase.model_validate(payload))
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc
    return cases


def dump_jsonl(path: Path, records: list[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        if isinstance(record, BaseModel):
            payload = record.model_dump()
        else:
            payload = record
        lines.append(json.dumps(payload, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def load_config(path: Path) -> BenchmarkConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    sections = {"weights", "hard_gates", "pricing", "judge_min_score", "judge_min_subscores", "runtime"}
    unknown_sections = data.keys() - sections
    if unknown_sections:
        raise ValueError("Unknown benchmark config sections: " + ", ".join(sorted(unknown_sections)))
    runtime = data.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a TOML table")
    unknown_runtime = runtime.keys() - {"judge_model", "judge_enabled", "request_timeout_seconds"}
    if unknown_runtime:
        raise ValueError("Unknown runtime config keys: " + ", ".join(sorted(unknown_runtime)))
    config_payload = {key: value for key, value in data.items() if key != "runtime"}
    return BenchmarkConfig.model_validate({**config_payload, **runtime})
