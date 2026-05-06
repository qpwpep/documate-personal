from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.core.answer_schema import ClaimItem
from src.core.contracts.debug import ActionResults, LLMCallMetadata, ModelUsageStatus, PlannerDiagnostic, RetrievalDiagnostic, TokenUsage
from src.core.evidence import EvidenceItem
from src.core.latency import LatencyBreakdownModel
from .config_models import CaseCategory, CaseScenario


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
    response_errors: list[str] = Field(default_factory=list)
    judge_errors: list[str] = Field(default_factory=list)
    judge_audit_failures: list[str] = Field(default_factory=list)
    action_results: ActionResults | None = None
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
    valid_claim_count: int = 0
    invalid_claim_count: int = 0
    section_count: int = 0
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
        if self.output_tokens <= 0 and self.token_usage is not None:
            self.output_tokens = int(self.token_usage.completion_tokens or 0)
        if self.section_count <= 0 and isinstance(self.response_payload, dict):
            sections = self.response_payload.get("sections")
            if isinstance(sections, list):
                self.section_count = len([item for item in sections if isinstance(item, dict)])
        if self.synthesis_mode is None and self.latency_breakdown and self.latency_breakdown.synthesis_attempts:
            self.synthesis_mode = self.latency_breakdown.synthesis_attempts[0].mode
        return self
