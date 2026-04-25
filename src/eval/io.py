from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config_models import BenchmarkCase, BenchmarkConfig


def load_cases_jsonl(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = line.strip()
        if not record:
            continue
        cases.append(BenchmarkCase.model_validate_json(record))
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

    config_payload: dict[str, Any] = {
        "weights": data.get("weights", {}),
        "hard_gates": data.get("hard_gates", {}),
        "pricing": data.get("pricing", {}),
        "judge_min_score": data.get("judge_min_score", {}),
        "judge_model": data.get("runtime", {}).get("judge_model", "gpt-5-mini"),
        "judge_enabled": data.get("runtime", {}).get("judge_enabled", True),
        "request_timeout_seconds": data.get("runtime", {}).get("request_timeout_seconds", 60),
    }
    return BenchmarkConfig(**config_payload)
