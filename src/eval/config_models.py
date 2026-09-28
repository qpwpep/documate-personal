from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.slack_contract import RecipientSelector, SlackDefault


CaseCategory = Literal["docs_only", "rag_only", "hybrid", "tool_action"]
CaseScenario = Literal[
    "seed_mutation", "adversarial", "regression", "ambiguity",
    "standard", "boundary", "injection", "correction", "failure",
]


class CaseWeightOverride(BaseModel):
    answer_quality: float | None = Field(default=None, ge=0.0)
    reference_coverage: float | None = Field(default=None, ge=0.0)
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


class SaveTarget(BaseModel):
    """The benchmark's oracle, chosen before observing an agent's response."""

    kind: Literal["final_answer", "setup_answer"]
    setup_turn_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_index(self) -> "SaveTarget":
        if (self.kind == "setup_answer") != (self.setup_turn_index is not None):
            raise ValueError("setup_answer requires setup_turn_index; final_answer must omit it")
        return self


class SaveExpectation(BaseModel):
    outcome: Literal["required_success", "expected_failure", "must_not_execute"]
    target: SaveTarget | None = None
    error_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expectation(self) -> "SaveExpectation":
        if self.outcome != "must_not_execute" and self.target is None:
            raise ValueError("a save expectation requires an explicit target")
        if self.outcome == "must_not_execute" and self.target is not None:
            raise ValueError("must_not_execute cannot declare a saved target")
        if (self.outcome == "expected_failure") != bool(self.error_codes):
            raise ValueError("only expected_failure requires nonempty error_codes")
        if any(not code.strip() for code in self.error_codes):
            raise ValueError("save error codes must be nonblank")
        return self


class OracleEvidence(BaseModel):
    """A fixed source excerpt supporting a case's expected answer or instruction."""

    source: str
    locator: str
    excerpt: str


class CaseOracle(BaseModel):
    """Semantic expectations; source excerpts are data, never judge instructions."""

    required_facts: list[str] = Field(default_factory=list)
    expected_behaviors: list[str] = Field(default_factory=list)
    forbidden_behaviors: list[str] = Field(default_factory=list)
    evidence: list[OracleEvidence] = Field(default_factory=list)
    ambiguity_resolution: str = ""


class BenchmarkCase(BaseModel):
    case_id: str
    category: CaseCategory
    scenario: CaseScenario = "seed_mutation"
    query: str
    setup_turns: list[str] = Field(default_factory=list)
    upload_fixtures: list[str] = Field(default_factory=list)
    upload_fixture: str | None = None
    slack_recipient: RecipientSelector | None = None
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    require_official_citation: bool = False
    require_local_citation: bool = False
    judge_rubric: str = ""
    judge_min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    weight_override: CaseWeightOverride | None = None
    save_expectation: SaveExpectation | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    evaluation_role: Literal["public_regression", "new_evaluation"] | None = None
    capability: str | None = None
    oracle: CaseOracle | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reject_obsolete_slack_fields(cls, value: Any) -> Any:
        # Authored extensions stay in the raw specification; recipient input
        # must never be silently ignored as one of those extensions.
        if isinstance(value, dict):
            obsolete = [key for key in ("slack_channel_id", "slack_user_id", "slack_email") if key in value]
            if obsolete:
                raise ValueError("Use slack_recipient instead of obsolete recipient fields: " + ", ".join(obsolete))
        return value

    @model_validator(mode="after")
    def validate_upload_declarations(self) -> "BenchmarkCase":
        if self.upload_fixture and self.upload_fixtures:
            raise ValueError("Use upload_fixtures or legacy upload_fixture, not both")
        target = self.save_expectation.target if self.save_expectation else None
        if target is not None and target.kind == "setup_answer" and target.setup_turn_index >= len(self.setup_turns):
            raise ValueError("save target setup_turn_index must select a declared setup turn")
        return self

    @property
    def resolved_upload_fixtures(self) -> list[str]:
        if self.upload_fixtures:
            return list(self.upload_fixtures)
        return [self.upload_fixture] if self.upload_fixture else []


class BenchmarkLiveSlackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    channel_id: str | None = None
    dm_recipient: RecipientSelector | None = None
    dm_default: SlackDefault = Field(default_factory=SlackDefault)

    @field_validator("channel_id")
    @classmethod
    def validate_channel(cls, value: str | None) -> str | None:
        return RecipientSelector(kind="channel", value=value).value if value is not None else None

    @field_validator("dm_recipient")
    @classmethod
    def validate_dm(cls, value: RecipientSelector | None) -> RecipientSelector | None:
        if value is not None and value.kind == "channel":
            raise ValueError("live Slack DM destination must be a user or email")
        return value

    @field_validator("dm_default")
    @classmethod
    def validate_dm_default(cls, value: SlackDefault) -> SlackDefault:
        if value.selector is not None and value.selector.kind == "channel":
            raise ValueError("live Slack DM default must be a user or email")
        return value

    def resolve_dm_recipient(self) -> RecipientSelector | None:
        if self.dm_recipient is not None:
            return self.dm_recipient
        if self.dm_default.failure is not None:
            raise ValueError(
                "Invalid app-level default Slack DM recipient: " + self.dm_default.failure.message
                + " Set exactly one of SLACK_DEFAULT_USER_ID / SLACK_DEFAULT_DM_EMAIL, "
                "or provide --live-slack-user-id / --live-slack-email."
            )
        return self.dm_default.selector

    def applies_to_case(self, case: BenchmarkCase) -> bool:
        return self.enabled and "slack_notify" in case.expected_tools

    def requires_channel_destination(self, case: BenchmarkCase) -> bool:
        return self.applies_to_case(case) and case.slack_recipient is not None and case.slack_recipient.kind == "channel"

    def requires_dm_destination(self, case: BenchmarkCase) -> bool:
        return self.applies_to_case(case) and not self.requires_channel_destination(case)

    def has_channel_destination(self) -> bool:
        return bool(self.channel_id)

    def has_dm_destination(self) -> bool:
        return self.resolve_dm_recipient() is not None


class ScoreWeights(BaseModel):
    answer_quality: float = 0.20
    reference_coverage: float = 0.20
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


class HardGates(BaseModel):
    pass_rate: float = 0.90
    tool_precision: float = 0.90
    tool_recall: float = 0.85
    citation_compliance: float = 0.95
    p95_latency_ms: int = 10000
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
    """Release gate for the judge's overall score, per category.

    These thresholds are policy starting points, not human-calibrated optima.
    A missing category must never silently remove the check, so every category
    keeps an explicit value.
    """

    docs_only: float = Field(default=0.70, ge=0.0, le=1.0)
    rag_only: float = Field(default=0.70, ge=0.0, le=1.0)
    hybrid: float = Field(default=0.70, ge=0.0, le=1.0)
    tool_action: float = Field(default=0.70, ge=0.0, le=1.0)

    def for_category(self, category: str) -> float | None:
        return getattr(self, str(category), None)


class JudgeSubscoreMinConfig(BaseModel):
    """Minimum subscore gates the semantic judge must satisfy.

    answer_quality applies to every category. groundedness applies to
    evidence-based answers; pure action cases are exempt because they do not
    require retrieval grounding.
    """

    answer_quality: float = Field(default=0.70, ge=0.0, le=1.0)
    groundedness: float = Field(default=0.70, ge=0.0, le=1.0)


class BenchmarkConfig(BaseModel):
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    hard_gates: HardGates = Field(default_factory=HardGates)
    pricing: Pricing = Field(default_factory=Pricing)
    judge_min_score: JudgeMinScoreConfig = Field(default_factory=JudgeMinScoreConfig)
    judge_min_subscores: JudgeSubscoreMinConfig = Field(default_factory=JudgeSubscoreMinConfig)
    judge_model: str = "gpt-5.6-luna"
    judge_enabled: bool = True
    request_timeout_seconds: int = 60
