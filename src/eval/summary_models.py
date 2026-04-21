from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


RunTrack = Literal["release", "smoke"]
PlannerErrorCode = Literal[
    "structured_output_invocation_failed",
    "output_validation_failed",
    "sanitized_output_validation_failed",
    "upload_route_dropped",
]


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
    track: RunTrack = "release"
    requested_limit: int | None = None
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

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_track_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if payload.get("track") is None:
            payload["track"] = "smoke" if payload.get("requested_limit") is not None else "release"
        return payload
