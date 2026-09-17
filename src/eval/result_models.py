from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.core.answer_schema import AnswerResponse, ActionReceipt
from src.core.contracts.debug import LLMCallMetadata, ModelUsageStatus, PlannerDiagnostic, RetrievalDiagnostic, TokenUsage
from src.core.contracts.provenance import AnswerProvenance
from src.core.evidence import EvidenceRef, SearchHit
from src.core.latency import LatencyBreakdownModel
from src.core.uploads import UploadManifest
from .config_models import CaseCategory, CaseScenario


JudgeStatus = Literal["disabled", "not_run", "failed", "succeeded", "legacy_unknown"]
EvalValidity = Literal["valid", "incomplete", "invalid", "legacy_unknown"]
# not_run: evaluation never reached the judge. failed: the judge boundary was
# invoked but its contract failed. The two must not share one bucket.
JudgeStatusReason = Literal[
    "missing_final_response",
    "input_incomplete",
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
    query: str
    request_payload: dict[str, Any]
    http_status: int = 0
    request_id: str | None = None
    response: AnswerResponse | None = None
    raw_final_response: dict[str, Any] | None = None
    trace: str | None = None
    debug: dict[str, Any] | None = None
    answer_provenance: AnswerProvenance | None = None
    evidence_assessment: EvidenceAssessment | None = None
    observed_hits: list[SearchHit] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
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
    run_id: str
    case_id: str
    category: CaseCategory
    scenario: CaseScenario = "seed_mutation"
    query: str
    session_id: str
    endpoint: str
    upload_fixture: str | None = None
    upload_fixtures: list[str] = Field(default_factory=list)
    attachment_fingerprints: dict[str, str] = Field(default_factory=dict)
    request_payload: dict[str, Any]
    request_id: str | None = None
    http_status: int
    response_text: str = ""
    response: AnswerResponse | None = None
    debug: dict[str, Any] | None = None
    answer_provenance: AnswerProvenance | None = None
    evidence_assessment: EvidenceAssessment | None = None
    observed_hits: list[SearchHit] = Field(default_factory=list)
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    planner_diagnostics: PlannerDiagnostic | None = None
    trace: str | None = None
    latency_ms_e2e: int | None = None
    attachment_setup_ms: int | None = Field(default=None, ge=0)
    question_response_ms: int | None = Field(default=None, ge=0)
    scenario_total_ms: int | None = Field(default=None, ge=0)
    scenario_turns: list[ScenarioTurnResult] = Field(default_factory=list)
    latency_ms_server: int | None = None
    latency_breakdown: LatencyBreakdownModel | None = None
    tool_calls: list[str] = Field(default_factory=list)
    token_usage: TokenUsage | None = None
    output_tokens: int = 0
    model_name: str | None = None
    models_used: list[str] = Field(default_factory=list)
    model_usage_status: ModelUsageStatus = "missing_debug"
    llm_calls: list[LLMCallMetadata] = Field(default_factory=list)
    tool_call_count: int = 0
    planner_errors: list[str] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    validation_events: list[str] = Field(default_factory=list)
    edge_decisions: list[dict[str, Any]] = Field(default_factory=list)
    debug_errors: list[str] = Field(default_factory=list)
    runtime_errors: list[str] = Field(default_factory=list)
    cleanup_errors: list[str] = Field(default_factory=list)
    response_errors: list[str] = Field(default_factory=list)
    judge_status: JudgeStatus | None = None
    judge_status_reason: JudgeStatusReason | None = None
    eval_validity: EvalValidity | None = None
    judge_input_issues: list[str] = Field(default_factory=list)
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
    judge_score_total: float | None = None
    llm_judge_score: float | None = None
    llm_judge_reason: str | None = None
    judge_input_complete: bool | None = None
    judge_min_score_applied: float | None = None
    judge_gate_passed: bool | None = None
    invalid_eval: bool = False
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
        # Records written before the explicit state contract carry no judge_status.
        # They stay readable as legacy data, but the mirrors below only run for
        # them; a new-contract record never lets an alias field outrank its
        # canonical verdict.
        if payload.get("judge_status") is not None:
            if payload.get("final_score") is None and payload.get("composite_quality_score") is not None:
                payload["final_score"] = payload.get("composite_quality_score")
            if payload.get("passed") is None and payload.get("release_pass") is not None:
                payload["passed"] = payload.get("release_pass")
            if payload.get("judge_gate_passed") is None and payload.get("judge_pass") is not None:
                payload["judge_gate_passed"] = payload.get("judge_pass")
        else:
            payload["judge_status"] = "legacy_unknown"
            payload["eval_validity"] = payload.get("eval_validity") or "legacy_unknown"
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
        judge_errors = payload.get("judge_errors")
        if isinstance(judge_errors, list):
            audit_failures = [
                str(item)
                for item in judge_errors
                if "judge_min_score audit failed" in str(item)
            ]
            if audit_failures:
                existing_audit_failures = payload.get("judge_audit_failures")
                merged_audit_failures = [
                    str(item)
                    for item in (existing_audit_failures if isinstance(existing_audit_failures, list) else [])
                    if str(item).strip()
                ]
                for item in audit_failures:
                    if item not in merged_audit_failures:
                        merged_audit_failures.append(item)
                payload["judge_audit_failures"] = merged_audit_failures
                payload["judge_errors"] = [
                    item
                    for item in judge_errors
                    if "judge_min_score audit failed" not in str(item)
                ]
        return payload

    @model_validator(mode="after")
    def mirror_legacy_result_fields(self) -> "CaseResult":
        if self.judge_status == "legacy_unknown":
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
        else:
            # New contract: the canonical verdict fields are authoritative and
            # alias mirrors always follow them, never the other way around.
            self.passed = self.release_pass
            self.final_score = self.composite_quality_score
            self.judge_gate_passed = self.judge_pass
            if self.eval_validity not in {"valid", "incomplete", "invalid"}:
                raise ValueError("new-contract results require an explicit eval_validity")
            if self.judge_status == "succeeded":
                if self.llm_judge_score is None or self.judge_subscores is None or self.judge_pass is None:
                    raise ValueError("a succeeded judge verdict requires score, subscores, and judge_pass")
            else:
                if self.llm_judge_score is not None or self.judge_subscores is not None or self.judge_pass is not None:
                    raise ValueError("a non-succeeded judge status cannot carry a score or verdict")
            if self.eval_validity in {"incomplete", "invalid"} and self.release_pass is not False:
                raise ValueError("incomplete or invalid evaluations cannot release")
            if self.release_pass is True and self.judge_pass is not True:
                raise ValueError("release requires a succeeded judge verdict that passed")
            if self.release_pass is True and self.product_pass is not True:
                raise ValueError("release requires a passing product verdict")
            if self.release_pass is True and self.save_assessment is not None and self.save_assessment.passed is False:
                raise ValueError("release requires the save outcome contract to pass")
            if self.release_pass is True and any(item.passed is False for item in self.setup_save_assessments.values()):
                raise ValueError("release requires setup saved artifacts to remain verified")
            if self.release_pass is True and self.save_assessment is not None and self.gate_failures:
                raise ValueError("release cannot carry blocking gate failures")
            if self.invalid_eval != (self.eval_validity == "invalid"):
                raise ValueError("invalid_eval must mirror eval_validity == 'invalid'")
        if self.tool_call_count <= 0 and self.tool_calls:
            self.tool_call_count = len(self.tool_calls)
        if self.output_tokens <= 0 and self.token_usage is not None:
            self.output_tokens = int(self.token_usage.completion_tokens or 0)
        if self.synthesis_mode is None and self.latency_breakdown and self.latency_breakdown.synthesis_attempts:
            self.synthesis_mode = self.latency_breakdown.synthesis_attempts[0].mode
        return self
