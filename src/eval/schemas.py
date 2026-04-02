from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..answer_schema import ClaimItem
from ..contracts.debug import LLMCallMetadata, PlannerDiagnostic, RetrievalDiagnostic, TokenUsage
from ..evidence import EvidenceItem
from ..latency import LatencyBreakdownModel


CaseCategory = Literal["docs_only", "rag_only", "hybrid", "tool_action"]
CaseScenario = Literal["seed_mutation", "adversarial", "regression", "ambiguity"]
PlannerErrorCode = Literal[
    "structured_output_invocation_failed",
    "output_validation_failed",
    "sanitized_output_validation_failed",
    "upload_route_dropped",
]


class CaseWeightOverride(BaseModel):
    answer_quality: float | None = Field(default=None, ge=0.0)
    groundedness: float | None = Field(default=None, ge=0.0)
    citation_traceability: float | None = Field(default=None, ge=0.0)
    tool_choice: float | None = Field(default=None, ge=0.0)
    format_language: float | None = Field(default=None, ge=0.0)
    llm_judge: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        legacy_map = {
            "tool_match": "tool_choice",
            "content_constraints": "answer_quality",
            "citation_compliance": "citation_traceability",
            "safety_format": "format_language",
        }
        for legacy_key, new_key in legacy_map.items():
            if new_key not in payload and legacy_key in payload:
                payload[new_key] = payload.get(legacy_key)
        return payload

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
    judge_min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    weight_override: CaseWeightOverride | None = None


class ScoreWeights(BaseModel):
    answer_quality: float = 0.20
    groundedness: float = 0.20
    citation_traceability: float = 0.20
    tool_choice: float = 0.15
    format_language: float = 0.05
    llm_judge: float = 0.20

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        legacy_map = {
            "tool_match": "tool_choice",
            "content_constraints": "answer_quality",
            "citation_compliance": "citation_traceability",
            "safety_format": "format_language",
        }
        for legacy_key, new_key in legacy_map.items():
            if new_key not in payload and legacy_key in payload:
                payload[new_key] = payload.get(legacy_key)
        return payload

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()

    @property
    def tool_match(self) -> float:
        return float(self.tool_choice)

    @property
    def content_constraints(self) -> float:
        return float(self.answer_quality)

    @property
    def citation_compliance(self) -> float:
        return float(self.citation_traceability)

    @property
    def safety_format(self) -> float:
        return float(self.format_language)


class HardGates(BaseModel):
    pass_rate: float = 0.90
    tool_precision: float = 0.90
    tool_recall: float = 0.85
    citation_compliance: float = 0.95
    p95_latency_ms: int = 20000
    avg_cost_per_case_usd: float = 0.01
    cost_gate_min_llm_call_coverage: float = 0.80


class ModelPricing(BaseModel):
    prompt_per_1k_usd: float
    completion_per_1k_usd: float


class Pricing(BaseModel):
    prompt_per_1k_usd: float = 0.00015
    completion_per_1k_usd: float = 0.0006
    models: dict[str, ModelPricing] = Field(default_factory=dict)


class JudgeMinScoreConfig(BaseModel):
    docs_only: float | None = Field(default=None, ge=0.0, le=1.0)
    hybrid: float | None = Field(default=None, ge=0.0, le=1.0)

    def for_category(self, category: str) -> float | None:
        return getattr(self, str(category), None)


class BenchmarkConfig(BaseModel):
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    hard_gates: HardGates = Field(default_factory=HardGates)
    pricing: Pricing = Field(default_factory=Pricing)
    judge_min_score: JudgeMinScoreConfig = Field(default_factory=JudgeMinScoreConfig)
    judge_model: str = "gpt-5-mini"
    judge_enabled: bool = True
    request_timeout_seconds: int = 60


class JudgeSubscores(BaseModel):
    answer_quality: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    citation_traceability: float = Field(ge=0.0, le=1.0)
    tool_choice: float = Field(ge=0.0, le=1.0)
    format_language: float = Field(ge=0.0, le=1.0)

    def average(self) -> float:
        values = self.model_dump().values()
        return sum(float(value) for value in values) / 5.0


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
    request_id: str | None = None
    http_status: int
    response_text: str = ""
    response_payload: dict[str, Any] | None = None
    response_claims: list[ClaimItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    observed_evidence: list[EvidenceItem] = Field(default_factory=list)
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    planner_diagnostics: PlannerDiagnostic | None = None
    file_path: str | None = None
    trace: str | None = None
    latency_ms_e2e: int | None = None
    latency_ms_server: int | None = None
    latency_breakdown: LatencyBreakdownModel | None = None
    tool_calls: list[str] = Field(default_factory=list)
    token_usage: TokenUsage | None = None
    model_name: str | None = None
    models_used: list[str] = Field(default_factory=list)
    llm_calls: list[LLMCallMetadata] = Field(default_factory=list)
    tool_call_count: int = 0
    planner_errors: list[str] = Field(default_factory=list)
    debug_errors: list[str] = Field(default_factory=list)
    runtime_errors: list[str] = Field(default_factory=list)
    response_errors: list[str] = Field(default_factory=list)
    judge_errors: list[str] = Field(default_factory=list)
    validator_reason: str | None = None
    validator_feedback: str | None = None
    effective_weights: dict[str, float] = Field(default_factory=dict)
    rule_scores: dict[str, float] = Field(default_factory=dict)
    rule_score_total: float | None = None
    debug_schema_version: int | None = None
    debug_observability_status: str | None = None
    missing_required_debug_fields: list[str] = Field(default_factory=list)
    judge_subscores: JudgeSubscores | None = None
    judge_score_total: float | None = None
    llm_judge_score: float | None = None
    llm_judge_reason: str | None = None
    judge_input_complete: bool | None = None
    judge_min_score_applied: float | None = None
    judge_gate_passed: bool | None = None
    invalid_eval: bool = False
    valid_claim_count: int = 0
    invalid_claim_count: int = 0
    synthesis_mode: str | None = None
    gate_failures: list[str] = Field(default_factory=list)
    composite_quality_score: float | None = None
    product_pass: bool | None = None
    judge_pass: bool | None = None
    release_pass: bool | None = None
    final_score: float | None = Field(default=None, exclude=True)
    passed: bool | None = Field(default=None, exclude=True)
    cost_usd: float | None = None
    created_at_utc: str

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_result_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if payload.get("composite_quality_score") is None and payload.get("final_score") is not None:
            payload["composite_quality_score"] = payload.get("final_score")
        if payload.get("final_score") is None and payload.get("composite_quality_score") is not None:
            payload["final_score"] = payload.get("composite_quality_score")
        if payload.get("release_pass") is None and payload.get("passed") is not None:
            payload["release_pass"] = payload.get("passed")
        if payload.get("passed") is None and payload.get("release_pass") is not None:
            payload["passed"] = payload.get("release_pass")
        if payload.get("product_pass") is None and payload.get("release_pass") is not None:
            payload["product_pass"] = payload.get("release_pass")
        if payload.get("judge_pass") is None and payload.get("judge_gate_passed") is not None:
            payload["judge_pass"] = payload.get("judge_gate_passed")
        if payload.get("judge_gate_passed") is None and payload.get("judge_pass") is not None:
            payload["judge_gate_passed"] = payload.get("judge_pass")
        if "response_claims" not in payload:
            response_payload = payload.get("response_payload")
            if isinstance(response_payload, dict) and isinstance(response_payload.get("claims"), list):
                payload["response_claims"] = response_payload.get("claims")
        return payload

    @model_validator(mode="after")
    def mirror_legacy_result_fields(self) -> "CaseResult":
        if self.composite_quality_score is None and self.final_score is not None:
            self.composite_quality_score = self.final_score
        if self.final_score is None and self.composite_quality_score is not None:
            self.final_score = self.composite_quality_score
        if self.release_pass is None and self.passed is not None:
            self.release_pass = self.passed
        if self.passed is None and self.release_pass is not None:
            self.passed = self.release_pass
        if self.product_pass is None and self.release_pass is not None:
            self.product_pass = self.release_pass
        if self.judge_pass is None and self.judge_gate_passed is not None:
            self.judge_pass = self.judge_gate_passed
        if self.judge_gate_passed is None and self.judge_pass is not None:
            self.judge_gate_passed = self.judge_pass
        if not self.response_claims and isinstance(self.response_payload, dict):
            claims = self.response_payload.get("claims")
            if isinstance(claims, list):
                try:
                    self.response_claims = [ClaimItem.model_validate(item) for item in claims]
                except Exception:
                    self.response_claims = []
        if self.tool_call_count <= 0 and self.tool_calls:
            self.tool_call_count = len(self.tool_calls)
        if self.synthesis_mode is None and self.latency_breakdown and self.latency_breakdown.synthesis_attempts:
            self.synthesis_mode = self.latency_breakdown.synthesis_attempts[0].mode
        return self


class SummaryStats(BaseModel):
    total_cases: int
    scored_cases: int
    passed_cases: int
    pass_rate: float
    product_passed_cases: int = 0
    judge_passed_cases: int = 0
    release_passed_cases: int = 0
    product_pass_rate: float = 0.0
    judge_pass_rate: float | None = None
    release_pass_rate: float = 0.0
    tool_precision: float
    tool_recall: float
    citation_compliance: float
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    avg_cost_per_case_usd: float | None = None
    cost_gate_eligible: bool = False
    llm_call_coverage_rate: float = 0.0
    request_id_coverage_rate: float = 0.0
    judge_input_completeness_rate: float | None = None
    judge_min_score_failures: int = 0
    deterministic_direct_usage_rate: float = 0.0
    high_rule_low_judge_divergence_rate: float = 0.0
    planner_deterministic_rate: float | None = None
    planner_llm_attempt_count: int | None = None
    planner_structured_success_rate: float | None = None
    synthesis_structured_success_rate: float | None = None
    failures: list[dict[str, str]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_summary_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if payload.get("release_passed_cases") is None and payload.get("passed_cases") is not None:
            payload["release_passed_cases"] = payload.get("passed_cases")
        if payload.get("release_pass_rate") is None and payload.get("pass_rate") is not None:
            payload["release_pass_rate"] = payload.get("pass_rate")
        return payload

    @model_validator(mode="after")
    def mirror_legacy_summary_fields(self) -> "SummaryStats":
        if self.release_passed_cases <= 0 and self.passed_cases:
            self.release_passed_cases = self.passed_cases
        if self.passed_cases <= 0 and self.release_passed_cases:
            self.passed_cases = self.release_passed_cases
        if self.release_pass_rate <= 0.0 and self.pass_rate:
            self.release_pass_rate = self.pass_rate
        if self.pass_rate <= 0.0 and self.release_pass_rate:
            self.pass_rate = self.release_pass_rate
        if self.product_passed_cases <= 0 and self.release_passed_cases:
            self.product_passed_cases = self.release_passed_cases
        if self.product_pass_rate <= 0.0 and self.release_pass_rate:
            self.product_pass_rate = self.release_pass_rate
        return self


class CategoryPassRate(BaseModel):
    category: str
    passed_cases: int
    total_cases: int
    pass_rate: float


class PlannerDiagnosticsBucket(BaseModel):
    category: str
    status: str
    reason: str | None = None
    override_reason: str | None = None
    count: int


class PlannerErrorBucket(BaseModel):
    category: str
    error_code: PlannerErrorCode
    count: int


class RetrievalRouteStatusBucket(BaseModel):
    category: str
    route: str
    status: str
    count: int


class RetrievalWarningBucket(BaseModel):
    category: str
    route: str
    warning: str
    count: int


class RouteConfusionBucket(BaseModel):
    category: str
    expected_routes: list[str] = Field(default_factory=list)
    observed_routes: list[str] = Field(default_factory=list)
    missing_expected_routes: list[str] = Field(default_factory=list)
    unexpected_routes: list[str] = Field(default_factory=list)
    forbidden_routes: list[str] = Field(default_factory=list)
    count: int


class ValidatorReasonBucket(BaseModel):
    category: str
    reason: str
    count: int
    share: float


class StageLatencyPercentile(BaseModel):
    stage: str
    sample_count: int
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None


class SynthesisModeBucket(BaseModel):
    category: str
    mode: str
    count: int


class LatencyBreakdownCoverage(BaseModel):
    available_cases: int = 0
    total_cases: int = 0
    coverage_rate: float = 0.0


class AnalysisStats(BaseModel):
    category_pass_rates: list[CategoryPassRate] = Field(default_factory=list)
    planner_diagnostics_histogram: list[PlannerDiagnosticsBucket] = Field(default_factory=list)
    planner_error_histogram: list[PlannerErrorBucket] = Field(default_factory=list)
    retrieval_route_status_histogram: list[RetrievalRouteStatusBucket] = Field(default_factory=list)
    retrieval_warning_histogram: list[RetrievalWarningBucket] = Field(default_factory=list)
    route_confusion: list[RouteConfusionBucket] = Field(default_factory=list)
    validator_reason_histogram: list[ValidatorReasonBucket] = Field(default_factory=list)
    synthesis_mode_histogram: list[SynthesisModeBucket] = Field(default_factory=list)
    stage_latency_percentiles: list[StageLatencyPercentile] = Field(default_factory=list)
    latency_breakdown_coverage: LatencyBreakdownCoverage | None = None


class GateResult(BaseModel):
    name: str
    threshold: float | int
    actual: float | int | None
    passed: bool
    gate_type: Literal["release", "audit"] = "release"
    detail: str | None = None
    status: str = "evaluated"


class RunSummary(BaseModel):
    run_id: str
    endpoint: str
    fixtures_path: str
    config_path: str
    generated_at_utc: str
    mode: str = "online"
    metrics: SummaryStats
    analysis: AnalysisStats | None = None
    gates: list[GateResult]
    overall_passed: bool
    weights: dict[str, float]
    hard_gates: dict[str, float | int]
    pricing: dict[str, Any]
    judge_enabled: bool
    judge_model: str
    audit_metrics: dict[str, Any] = Field(default_factory=dict)


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
        "judge_min_score": data.get("judge_min_score", {}),
        "judge_model": data.get("runtime", {}).get("judge_model", "gpt-5-mini"),
        "judge_enabled": data.get("runtime", {}).get("judge_enabled", True),
        "request_timeout_seconds": data.get("runtime", {}).get("request_timeout_seconds", 60),
    }
    return BenchmarkConfig(**config_payload)
