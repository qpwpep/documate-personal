from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.answer_schema import AnswerResponse, ActionReceipt
from src.core.contracts.outcome import TurnResult
from src.core.contracts.debug import PlannerDiagnostic, RetrievalDiagnostic
from src.core.contracts.usage import LLMCallRecord
from src.core.contracts.provenance import AnswerProvenance
from src.core.contracts.routing import RoutingDecision
from src.core.contracts.tool_execution import ToolExecutionEvidence
from src.core.evidence import EvidenceRef, SearchHit
from src.core.latency import LatencyBreakdownModel
from src.core.uploads import UploadManifest
from .config_models import CaseCategory, CaseScenario
from .decisions import CaseDecision
from .tool_policy import ToolPolicyAssessment, ToolPolicySpec
from .action_outcomes import ActionAssessment


JudgeStatus = Literal["disabled", "not_run", "failed", "succeeded"]
EvalValidity = Literal["valid", "incomplete", "invalid"]
# not_run: evaluation never reached the judge. failed: the judge boundary was
# invoked but its contract failed. The two must not share one bucket.
JudgeStatusReason = Literal[
    "product_failure",
    "missing_final_response",
    "input_incomplete",
    "input_budget_exceeded",
    "input_invalid",
    "client_unavailable",
    "invocation_failed",
    "output_invalid",
]


class JudgeSubscores(BaseModel):
    answer_quality: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    citation_traceability: float = Field(ge=0.0, le=1.0)
    tool_choice: float = Field(ge=0.0, le=1.0)
    format_language: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def reject_non_numeric_scores(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        for key, item in value.items():
            if item is None:
                continue
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise ValueError(f"judge subscore '{key}' must be a finite number, got {item!r}")
        return value

    def average(self) -> float:
        values = self.model_dump().values()
        return sum(float(value) for value in values) / 5.0


class EvidenceAssessment(BaseModel):
    status: Literal["complete", "unavailable", "invalid"]
    errors: list[str] = Field(default_factory=list)
    verified_evidence: list[EvidenceRef] = Field(default_factory=list)


class ScenarioTurnResult(BaseModel):
    role: Literal["setup", "question"] = "question"
    llm_calls: list[LLMCallRecord] | None = None
    query: str
    request_payload: dict[str, Any]
    http_status: int = 0
    execution_id: str | None = None
    response: AnswerResponse | None = None
    turn_result: TurnResult | None = Field(default=None, exclude_if=lambda value: value is None)
    raw_final_response: dict[str, Any] | None = None
    trace: str | None = None
    debug: dict[str, Any] | None = None
    answer_provenance: AnswerProvenance | None = None
    evidence_assessment: EvidenceAssessment | None = None
    observed_hits: list[SearchHit] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    execution_evidence: ToolExecutionEvidence | dict[str, Any] | None = None
    upload_manifest: UploadManifest | None = None
    question_response_ms: int | None = Field(default=None, ge=0)
    runtime_errors: list[str] = Field(default_factory=list)
    response_errors: list[str] = Field(default_factory=list)


class SaveAssessment(BaseModel):
    """An independent observation of a requested artifact, not a tool assertion."""

    status: Literal["verified", "failed", "unverifiable", "not_applicable"]
    outcome: Literal["required_success", "expected_failure", "must_not_execute"] | None = None
    passed: bool | None = None
    failure_codes: list[str] = Field(default_factory=list)
    expected_sha256: str | None = None
    expected_byte_count: int | None = None
    artifact_id: str | None = None
    observed_sha256: str | None = None
    observed_byte_count: int | None = None
    checked_at_utc: str | None = None
    phase: Literal["case", "run_end"] = "case"

    @model_validator(mode="after")
    def consistent_verdict(self) -> "SaveAssessment":
        if self.status == "not_applicable":
            if self.passed is not None or self.failure_codes or self.outcome is not None:
                raise ValueError("inapplicable saves cannot carry a verdict or an obligation")
        elif self.status == "verified":
            if self.passed is not True or self.failure_codes or self.outcome is None:
                raise ValueError("verified saves require a passing, failure-free declared outcome")
        elif self.passed is not False or not self.failure_codes:
            raise ValueError("failed or unverifiable saves require a failing verdict and reasons")
        return self


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_contract_version: Literal[3]
    policy_snapshot: ToolPolicySpec | None = None
    policy_assessment: ToolPolicyAssessment | None = None
    action_assessments: dict[str, ActionAssessment] = Field(default_factory=dict)
    decision: CaseDecision | None = None
    execution_evidence: ToolExecutionEvidence | dict[str, Any] | None = None
    run_id: str
    case_id: str
    category: CaseCategory
    scenario: CaseScenario = "seed_mutation"
    query: str
    session_id: str
    endpoint: str
    upload_fixtures: list[str] = Field(default_factory=list)
    attachment_fingerprints: dict[str, str] = Field(default_factory=dict)
    request_payload: dict[str, Any]
    execution_id: str | None = None
    http_status: int
    response_text: str = ""
    response: AnswerResponse | None = None
    turn_result: TurnResult | None = Field(default=None, exclude_if=lambda value: value is None)
    debug: dict[str, Any] | None = None
    answer_provenance: AnswerProvenance | None = None
    evidence_assessment: EvidenceAssessment | None = None
    observed_hits: list[SearchHit] = Field(default_factory=list)
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    planner_diagnostics: PlannerDiagnostic | None = None
    trace: str | None = None
    attachment_setup_ms: int | None = Field(default=None, ge=0)
    question_response_ms: int | None = Field(default=None, ge=0)
    scenario_total_ms: int | None = Field(default=None, ge=0)
    scenario_turns: list[ScenarioTurnResult] = Field(default_factory=list)
    latency_ms_server: int | None = None
    latency_breakdown: LatencyBreakdownModel | None = None
    tool_calls: list[str] = Field(default_factory=list)
    synthesis_output_tokens: int | None = Field(default=None, ge=0, strict=True)
    tool_call_count: int = 0
    planner_errors: list[str] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    validation_events: list[str] = Field(default_factory=list)
    route_decisions: list[RoutingDecision] = Field(default_factory=list)
    memory_compactions: list[dict[str, Any]] = Field(default_factory=list)
    debug_errors: list[str] = Field(default_factory=list)
    runtime_errors: list[str] = Field(default_factory=list)
    cleanup_errors: list[str] = Field(default_factory=list)
    response_errors: list[str] = Field(default_factory=list)
    judge_status: JudgeStatus
    judge_status_reason: JudgeStatusReason | None = None
    eval_validity: EvalValidity
    judge_input_issues: list[str] = Field(default_factory=list)
    judge_input_stats: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    judge_errors: list[str] = Field(default_factory=list)
    judge_audit_failures: list[str] = Field(default_factory=list)
    actions: list[ActionReceipt] = Field(default_factory=list)
    save_assessment: SaveAssessment | None = None
    setup_save_assessments: dict[int, SaveAssessment] = Field(default_factory=dict)
    slack_delivery_status: Literal["success", "failed", "skipped", "unknown", "not_applicable"] = "not_applicable"
    slack_delivery_required: bool = False
    slack_delivery_error: str | None = None
    validator_reason: str | None = None
    validator_feedback: str | None = None
    effective_weights: dict[str, float] = Field(default_factory=dict)
    rule_scores: dict[str, float] = Field(default_factory=dict)
    rule_score_total: float | None = None
    debug_schema_version: int | None = None
    debug_observability_status: str | None = None
    missing_required_debug_fields: list[str] = Field(default_factory=list)
    judge_subscores: JudgeSubscores | None = None
    llm_judge_score: float | None = None
    llm_judge_reason: str | None = None
    judge_input_complete: bool | None = None
    judge_min_score_applied: float | None = None
    resolved_unit_count: int = 0
    missing_reference_unit_count: int = 0
    unchecked_unit_count: int = 0
    exact_match_unit_count: int = 0
    unsupported_unit_count: int = 0
    block_count: int = 0
    synthesis_mode: str | None = None
    gate_failures: list[str] = Field(default_factory=list)
    composite_quality_score: float | None = None
    product_pass: bool | None = None
    judge_pass: bool | None = None
    release_pass: bool | None = None
    cost_usd: float | None = None
    created_at_utc: str

    @property
    def llm_calls(self) -> list[LLMCallRecord] | None:
        question = next((turn for turn in reversed(self.scenario_turns) if turn.role == "question"), None)
        return question.llm_calls if question is not None else None

    @model_validator(mode="after")
    def validate_judge_outcome(self) -> "CaseResult":
        if self.judge_status_reason == "product_failure" and (
            self.judge_status != "not_run" or self.product_pass is not False or self.eval_validity != "valid"
            or self.composite_quality_score is not None or self.judge_input_complete is not None
        ):
            raise ValueError("a product_failure judge skip requires a valid, unscored product failure")
        if self.judge_status == "succeeded":
            if self.llm_judge_score is None or self.judge_subscores is None or self.judge_pass is None:
                raise ValueError("a succeeded judge verdict requires score, subscores, and judge_pass")
        elif self.llm_judge_score is not None or self.judge_subscores is not None or self.judge_pass is not None:
            raise ValueError("a non-succeeded judge status cannot carry a score or verdict")
        return self
