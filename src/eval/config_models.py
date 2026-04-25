from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


CaseCategory = Literal["docs_only", "rag_only", "hybrid", "tool_action"]
CaseScenario = Literal["seed_mutation", "adversarial", "regression", "ambiguity"]
ExpectedBehavior = Literal[
    "answer",
    "needs_clarification",
    "insufficient_evidence",
    "cannot_verify",
    "handled_tool_failure",
    "refuse",
]
PlannerMode = Literal["auto", "force_llm"]
BENCHMARK_FIXTURE_SCHEMA_VERSION = 2


class GoldenFact(BaseModel):
    id: str
    description: str
    acceptable_terms: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_string_fact(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return {"id": text[:40] or "fact", "description": text, "acceptable_terms": [text]}
        return value


class GoldenCriteria(BaseModel):
    required_facts: list[GoldenFact] = Field(default_factory=list)
    acceptable_phrasings: list[str] = Field(default_factory=list)
    critical_errors: list[str] = Field(default_factory=list)


class BenchmarkStep(BaseModel):
    step_id: str | None = None
    query: str
    upload_fixture: str | None = None
    upload_fixtures: list[str] | None = None
    clear_uploads: bool = False
    slack_channel_id: str | None = None
    slack_user_id: str | None = None
    slack_email: str | None = None
    reset_slack_destination: bool = False
    expected_tools: list[str] | None = None
    forbidden_tools: list[str] | None = None
    must_include: list[str] | None = None
    must_not_include: list[str] | None = None
    require_official_citation: bool | None = None
    require_local_citation: bool | None = None
    golden_criteria: GoldenCriteria | None = None
    expected_behavior: ExpectedBehavior | None = None
    expected_error_codes: list[str] = Field(default_factory=list)
    planner_mode: PlannerMode | None = None
    faults: dict[str, str] | None = None


class CaseWeightOverride(BaseModel):
    answer_quality: float | None = Field(default=None, ge=0.0)
    criteria_coverage: float | None = Field(default=None, ge=0.0)
    groundedness: float | None = Field(default=None, ge=0.0)
    citation_traceability: float | None = Field(default=None, ge=0.0)
    tool_choice: float | None = Field(default=None, ge=0.0)
    uncertainty_handling: float | None = Field(default=None, ge=0.0)
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
    schema_version: int = BENCHMARK_FIXTURE_SCHEMA_VERSION
    benchmark_fixture_schema_version: int = BENCHMARK_FIXTURE_SCHEMA_VERSION
    case_id: str
    category: CaseCategory
    scenario: CaseScenario = "seed_mutation"
    query: str
    upload_fixture: str | None = None
    upload_fixtures: list[str] = Field(default_factory=list)
    slack_channel_id: str | None = None
    slack_user_id: str | None = None
    slack_email: str | None = None
    reset_slack_destination: bool = False
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    require_official_citation: bool = False
    require_local_citation: bool = False
    golden_criteria: GoldenCriteria = Field(default_factory=GoldenCriteria)
    expected_behavior: ExpectedBehavior = "answer"
    expected_error_codes: list[str] = Field(default_factory=list)
    steps: list[BenchmarkStep] = Field(default_factory=list)
    planner_mode: PlannerMode = "auto"
    faults: dict[str, str] = Field(default_factory=dict)
    judge_rubric: str = ""
    judge_min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    weight_override: CaseWeightOverride | None = None

    @model_validator(mode="after")
    def normalize_v2_fields(self) -> "BenchmarkCase":
        if self.upload_fixture and not self.upload_fixtures:
            self.upload_fixtures = [self.upload_fixture]
        if self.upload_fixtures and not self.upload_fixture:
            self.upload_fixture = self.upload_fixtures[0]
        if not self.golden_criteria.required_facts and self.must_include:
            self.golden_criteria.required_facts = [
                GoldenFact(
                    id=f"must_include_{index}",
                    description=needle,
                    acceptable_terms=[needle],
                )
                for index, needle in enumerate(self.must_include, start=1)
            ]
        return self


class BenchmarkLiveSlackConfig(BaseModel):
    enabled: bool = False
    channel_id: str | None = None
    user_id: str | None = None
    email: str | None = None
    fallback_user_id: str | None = None
    fallback_email: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_blank_values(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        for key in ("channel_id", "user_id", "email", "fallback_user_id", "fallback_email"):
            item = payload.get(key)
            if item is None:
                continue
            text = str(item).strip()
            payload[key] = text or None
        return payload

    def applies_to_case(self, case: BenchmarkCase) -> bool:
        return (
            self.enabled
            and "slack_notify" in case.expected_tools
            and "SLACK_DESTINATION_MISSING" not in case.expected_error_codes
        )

    def requires_channel_destination(self, case: BenchmarkCase) -> bool:
        return self.applies_to_case(case) and bool(case.slack_channel_id)

    def requires_dm_destination(self, case: BenchmarkCase) -> bool:
        return self.applies_to_case(case) and not self.requires_channel_destination(case)

    def has_channel_destination(self) -> bool:
        return bool(self.channel_id)

    def has_dm_destination(self) -> bool:
        return bool(self.user_id or self.email or self.fallback_user_id or self.fallback_email)

    def resolve_dm_payload(self) -> dict[str, str]:
        resolved_user_id = self.user_id or self.fallback_user_id
        if resolved_user_id:
            return {"slack_user_id": resolved_user_id}
        resolved_email = self.email or self.fallback_email
        if resolved_email:
            return {"slack_email": resolved_email}
        return {}


class ScoreWeights(BaseModel):
    answer_quality: float = 0.15
    criteria_coverage: float = 0.20
    groundedness: float = 0.17
    citation_traceability: float = 0.14
    tool_choice: float = 0.12
    uncertainty_handling: float = 0.10
    format_language: float = 0.04
    llm_judge: float = 0.08

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
