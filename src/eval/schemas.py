from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..evidence import EvidenceItem


CaseCategory = Literal["docs_only", "rag_only", "hybrid", "tool_action"]
CaseScenario = Literal["seed_mutation", "adversarial", "regression", "ambiguity"]


class CaseWeightOverride(BaseModel):
    tool_match: float | None = Field(default=None, ge=0.0)
    content_constraints: float | None = Field(default=None, ge=0.0)
    citation_compliance: float | None = Field(default=None, ge=0.0)
    safety_format: float | None = Field(default=None, ge=0.0)
    llm_judge: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_finite(self) -> "CaseWeightOverride":
        for key, value in self.model_dump(exclude_none=True).items():
            if not math.isfinite(float(value)):
                raise ValueError(f"weight_override.{key} must be a finite number")
        return self

    def as_partial_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.model_dump(exclude_none=True).items()}


class BenchmarkCase(BaseModel):
    case_id: str
    category: CaseCategory
    scenario: CaseScenario = "seed_mutation"
    query: str
    upload_fixture: str | None = None
    slack_channel_id: str | None = None
    slack_user_id: str | None = None
    slack_email: str | None = None
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    require_official_citation: bool = False
    require_local_citation: bool = False
    judge_rubric: str = ""
    weight_override: CaseWeightOverride | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ScoreWeights(BaseModel):
    tool_match: float = 0.30
    content_constraints: float = 0.25
    citation_compliance: float = 0.20
    safety_format: float = 0.05
    llm_judge: float = 0.20

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class HardGates(BaseModel):
    pass_rate: float = 0.82
    tool_precision: float = 0.90
    tool_recall: float = 0.85
    citation_compliance: float = 0.88
    p95_latency_ms: int = 20000
    avg_cost_per_case_usd: float = 0.035


class Pricing(BaseModel):
    prompt_per_1k_usd: float = 0.00015
    completion_per_1k_usd: float = 0.0006


class BenchmarkConfig(BaseModel):
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    hard_gates: HardGates = Field(default_factory=HardGates)
    pricing: Pricing = Field(default_factory=Pricing)
    judge_model: str = "gpt-5-mini"
    judge_enabled: bool = True
    request_timeout_seconds: int = 60


class CaseResult(BaseModel):
    run_id: str
    case_id: str
    category: CaseCategory
    scenario: CaseScenario = "seed_mutation"
    query: str
    session_id: str
    endpoint: str
    upload_fixture: str | None = None
    request_payload: dict[str, Any]
    http_status: int
    response_text: str = ""
    response_payload: dict[str, Any] | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    observed_evidence: list[EvidenceItem] = Field(default_factory=list)
    file_path: str | None = None
    trace: str | None = None
    latency_ms_e2e: int | None = None
    latency_ms_server: int | None = None
    tool_calls: list[str] = Field(default_factory=list)
    token_usage: TokenUsage | None = None
    model_name: str | None = None
    runtime_errors: list[str] = Field(default_factory=list)
    response_errors: list[str] = Field(default_factory=list)
    judge_errors: list[str] = Field(default_factory=list)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    rule_scores: dict[str, float] = Field(default_factory=dict)
    rule_score_total: float | None = None
    llm_judge_score: float | None = None
    llm_judge_reason: str | None = None
    final_score: float | None = None
    passed: bool | None = None
    cost_usd: float | None = None
    created_at_utc: str


class SummaryStats(BaseModel):
    total_cases: int
    scored_cases: int
    passed_cases: int
    pass_rate: float
    tool_precision: float
    tool_recall: float
    citation_compliance: float
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    avg_cost_per_case_usd: float | None = None
    failures: list[dict[str, str]] = Field(default_factory=list)


class GateResult(BaseModel):
    name: str
    threshold: float | int
    actual: float | int | None
    passed: bool


class RunSummary(BaseModel):
    run_id: str
    endpoint: str
    fixtures_path: str
    config_path: str
    generated_at_utc: str
    mode: str = "online"
    metrics: SummaryStats
    gates: list[GateResult]
    overall_passed: bool
    weights: dict[str, float]
    hard_gates: dict[str, float | int]
    pricing: dict[str, float]
    judge_enabled: bool
    judge_model: str


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
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_config(path: Path) -> BenchmarkConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    config_payload: dict[str, Any] = {
        "weights": data.get("weights", {}),
        "hard_gates": data.get("hard_gates", {}),
        "pricing": data.get("pricing", {}),
        "judge_model": data.get("runtime", {}).get("judge_model", "gpt-5-mini"),
        "judge_enabled": data.get("runtime", {}).get("judge_enabled", True),
        "request_timeout_seconds": data.get("runtime", {}).get("request_timeout_seconds", 60),
    }
    return BenchmarkConfig(**config_payload)
