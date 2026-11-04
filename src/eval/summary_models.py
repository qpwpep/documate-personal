from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from .decisions import ReleaseDecision, decide_release
from .tool_policy import ToolPolicySpec


RunTrack = Literal["release", "smoke"]
PlannerErrorCode = Literal[
    "PLANNER_SCHEMA_INVALID",
    "structured_output_invocation_failed",
    "output_validation_failed",
    "sanitized_output_validation_failed",
    "upload_route_dropped",
]


class SummaryStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_compliant_cases: int
    policy_violating_cases: int
    policy_indeterminate_cases: int
    policy_violations: int
    policy_failure_codes: list[str]
    total_cases: int
    scored_cases: int
    planned_cases: int = 0
    invalid_eval_cases: int = 0
    incomplete_eval_cases: int = 0
    missing_result_cases: int = 0
    duplicate_result_cases: int = 0
    unexpected_result_cases: int = 0
    judge_required_cases: int = 0
    judge_succeeded_cases: int = 0
    judge_failed_cases: int = 0
    judge_disabled_cases: int = 0
    judge_not_run_cases: int = 0
    judge_execution_rate: float | None = None
    product_passed_cases: int = 0
    judge_passed_cases: int = 0
    release_passed_cases: int = 0
    product_pass_rate: float = 0.0
    judge_pass_rate: float | None = None
    release_pass_rate: float = 0.0
    save_contract_cases: int = 0
    save_contract_failures: int = 0
    save_verified_cases: int = 0
    save_unverifiable_cases: int = 0
    action_contract_failures: int = 0
    action_unverifiable_cases: int = 0
    tool_precision: float
    tool_recall: float
    citation_compliance: float
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p50_attachment_setup_ms: float | None = None
    p95_attachment_setup_ms: float | None = None
    p50_question_response_ms: float | None = None
    p95_question_response_ms: float | None = None
    p50_scenario_total_ms: float | None = None
    p95_scenario_total_ms: float | None = None
    hybrid_p95_latency_ms: float | None = None
    hybrid_p95_server_ms: float | None = None
    hybrid_p95_synthesis_ms: float | None = None
    hybrid_p95_retrieval_ms: float | None = None
    docs_only_p95_latency_ms: float | None = None
    avg_cost_per_case_usd: float | None = None
    slack_delivery_required_cases: int = 0
    slack_delivery_success_cases: int = 0
    slack_delivery_success_rate: float | None = None
    cost_gate_eligible: bool = False
    cost_observation_rate: float | None = None
    cost_observed_cases: int | None = None
    execution_id_coverage_rate: float = 0.0
    judge_input_completeness_rate: float | None = None
    judge_min_score_failures: int = 0
    high_rule_low_judge_divergence_rate: float = 0.0
    planner_deterministic_rate: float | None = None
    planner_llm_attempt_count: int | None = None
    planner_structured_success_rate: float | None = None
    planner_error_count: int = 0
    planner_error_case_count: int = 0
    planner_warning_count: int = 0
    planner_duplicate_route_merge_count: int = 0
    planner_final_success_rate: float | None = None
    synthesis_structured_success_rate: float | None = None
    failures: list[dict[str, str]] = Field(default_factory=list)



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


class ErrorCodeBucket(BaseModel):
    category: str
    error_code: str
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


class SlackDeliveryStatusBucket(BaseModel):
    category: str
    status: str
    count: int


class LatencyBreakdownCoverage(BaseModel):
    available_cases: int = 0
    total_cases: int = 0
    coverage_rate: float = 0.0


class AnalysisStats(BaseModel):
    category_pass_rates: list[CategoryPassRate] = Field(default_factory=list)
    planner_diagnostics_histogram: list[PlannerDiagnosticsBucket] = Field(default_factory=list)
    planner_error_histogram: list[PlannerErrorBucket] = Field(default_factory=list)
    error_code_histogram: list[ErrorCodeBucket] = Field(default_factory=list)
    retrieval_route_status_histogram: list[RetrievalRouteStatusBucket] = Field(default_factory=list)
    retrieval_warning_histogram: list[RetrievalWarningBucket] = Field(default_factory=list)
    route_confusion: list[RouteConfusionBucket] = Field(default_factory=list)
    validator_reason_histogram: list[ValidatorReasonBucket] = Field(default_factory=list)
    synthesis_mode_histogram: list[SynthesisModeBucket] = Field(default_factory=list)
    slack_delivery_status_histogram: list[SlackDeliveryStatusBucket] = Field(default_factory=list)
    stage_latency_percentiles: list[StageLatencyPercentile] = Field(default_factory=list)
    latency_breakdown_coverage: LatencyBreakdownCoverage | None = None


class GateResult(BaseModel):
    name: str
    threshold: float | int | None
    actual: float | int | None
    passed: bool
    gate_type: Literal["release", "audit"] = "release"
    detail: str | None = None
    status: str = "evaluated"


class ReleaseSuiteContract(BaseModel):
    """The complete approved input, retained independently of selected cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approved_case_ids: tuple[str, ...] = Field(min_length=1)
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_case_ids(self) -> "ReleaseSuiteContract":
        if (any(not case_id.strip() for case_id in self.approved_case_ids)
                or len(set(self.approved_case_ids)) != len(self.approved_case_ids)):
            raise ValueError("approved suite case identities must be nonblank and unique")
        return self


def release_suite_gate(
    *, track: RunTrack, requested_limit: int | None, planned_ids: set[str],
    release_suite: ReleaseSuiteContract | None,
) -> GateResult:
    if track != "release":
        return GateResult(name="release_suite_completeness", threshold=0, actual=0, passed=True,
                          status="not_applicable", detail="Diagnostic scope; no release verdict.")
    if release_suite is None:
        return GateResult(name="release_suite_completeness", threshold=0, actual=1, passed=False,
                          status="evaluation_incomplete", detail="The approved full suite is not retained.")
    approved_ids = set(release_suite.approved_case_ids)
    missing = approved_ids - planned_ids
    unexpected = planned_ids - approved_ids
    issues = len(missing) + len(unexpected) + int(requested_limit is not None)
    return GateResult(
        name="release_suite_completeness", threshold=0, actual=issues, passed=issues == 0,
        status="evaluated" if not issues else "evaluation_incomplete",
        detail=(f"approved={len(approved_ids)} planned={len(planned_ids)} "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"limited={requested_limit is not None}"),
    )


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_contract_version: Literal[2]
    case_policy_snapshots: dict[str, ToolPolicySpec]
    results_fingerprint: str
    release_decision: ReleaseDecision
    run_id: str
    endpoint: str
    fixtures_path: str
    config_path: str
    generated_at_utc: str
    mode: str = "online"
    track: RunTrack
    requested_limit: int | None = None
    release_suite: ReleaseSuiteContract | None = None
    execution_contract_version: str | None = None
    measurement_contract_version: str | None = None
    suite_fingerprint: str | None = None
    evaluation_fingerprint: str | None = None
    metrics: SummaryStats
    analysis: AnalysisStats | None = None
    gates: list[GateResult]
    overall_passed: bool
    weight_profiles: dict[str, dict[str, float]]
    hard_gates: dict[str, float | int | None]
    pricing: dict[str, Any]
    judge_enabled: bool
    judge_model: str
    audit_metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_current_decision(self) -> "RunSummary":
        suite_gate = release_suite_gate(
            track=self.track, requested_limit=self.requested_limit,
            planned_ids=set(self.case_policy_snapshots), release_suite=self.release_suite,
        )
        if next((gate for gate in self.gates if gate.name == suite_gate.name), None) != suite_gate:
            raise ValueError("release suite gate must follow the retained full approved suite")
        if (self.metrics.policy_compliant_cases + self.metrics.policy_violating_cases
                + self.metrics.policy_indeterminate_cases != self.metrics.total_cases):
            raise ValueError("policy measurements must account for every observed case")
        policy_gate = next((gate for gate in self.gates if gate.name == "tool_execution_policy"), None)
        policy_pass = not (self.metrics.policy_violating_cases or self.metrics.policy_indeterminate_cases
                           or self.metrics.policy_failure_codes)
        if (policy_gate is None or policy_gate.gate_type != "release" or policy_gate.passed != policy_pass
                or policy_gate.actual != self.metrics.policy_violating_cases + self.metrics.policy_indeterminate_cases):
            raise ValueError("current runs require a consistent execution policy gate")
        completeness = (self.metrics.invalid_eval_cases + self.metrics.incomplete_eval_cases
                        + self.metrics.missing_result_cases + self.metrics.duplicate_result_cases
                        + self.metrics.unexpected_result_cases)
        for name, actual in (("evaluation_completeness", completeness),
                             ("save_outcome_contract", self.metrics.save_contract_failures),
                             ("action_outcome_contract", self.metrics.action_contract_failures)):
            gate = next((gate for gate in self.gates if gate.name == name), None)
            if gate is None or gate.gate_type != "release" or gate.actual != actual or gate.passed != (actual == 0):
                raise ValueError(f"inconsistent mandatory gate: {name}")
        cost_rate = ((self.metrics.cost_observed_cases or 0) / self.metrics.planned_cases
                     if self.metrics.planned_cases else 0.0)
        eligible = (bool(self.metrics.cost_observed_cases)
                    and cost_rate >= self.hard_gates["cost_gate_min_observation_rate"])
        cost_observation = next((gate for gate in self.gates if gate.name == "cost_observation_rate"), None)
        cost = next((gate for gate in self.gates if gate.name == "avg_cost_per_case_usd"), None)
        cost_passed = (eligible and self.metrics.avg_cost_per_case_usd is not None
                       and self.metrics.avg_cost_per_case_usd <= self.hard_gates["avg_cost_per_case_usd"])
        if (self.metrics.cost_observation_rate != round(cost_rate, 4)
                or not 0 <= (self.metrics.cost_observed_cases or 0) <= self.metrics.planned_cases
                or self.metrics.cost_gate_eligible != eligible
                or (self.metrics.avg_cost_per_case_usd is not None
                    and (not math.isfinite(self.metrics.avg_cost_per_case_usd)
                         or self.metrics.avg_cost_per_case_usd < 0))
                or cost_observation is None or cost_observation.gate_type != "release"
                or cost_observation.actual != self.metrics.cost_observation_rate
                or cost_observation.threshold != self.hard_gates["cost_gate_min_observation_rate"]
                or cost_observation.passed != eligible
                or cost_observation.status != ("evaluated" if eligible else "observation_insufficient")
                or cost is None or cost.gate_type != "release" or cost.passed != cost_passed
                or cost.threshold != self.hard_gates["avg_cost_per_case_usd"]
                or cost.status != ("evaluated" if eligible else "observation_insufficient")
                or cost.actual != (self.metrics.avg_cost_per_case_usd if eligible else None)):
            raise ValueError("cost gates must follow the retained cost observations")
        expected = decide_release(gates=self.gates, track=self.track,
                                  policy_failure_codes=self.metrics.policy_failure_codes)
        if self.release_decision != expected or self.overall_passed != expected.passed:
            raise ValueError("release decision must follow the central run decision")
        return self
