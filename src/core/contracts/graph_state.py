from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypedDict

from src.core.answer_schema import ActionReceipt, AnswerResponse
from src.core.save_contract import SaveOperation
from src.core.slack_contract import RecipientSelector, SlackDelivery
from src.core.evidence import EvidenceRef, SearchHit
from src.core.decisions import prepared_packet_fingerprint
from src.core.planner_schema import PlannerOutput
from src.core.llm_errors import ExecutionProblem, LLMDiagnostic
from src.core.request_contracts import RequestContract, UserTurnSnapshot
from src.core.uploads import UploadFileInfo
from src.core.contracts.debug import DebugDiagnostics, PlannerDiagnostic, PlannerStatus, RetryState, RetrievalDiagnostic, empty_planner_diagnostic
from src.core.contracts.provenance import AnswerSource, BodyKind
from src.core.contracts.routing import RoutingDecision


class SessionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    slack_recipient: RecipientSelector | None = None


class PendingAction(BaseModel):
    """An unfinished request owns its requirements and any checked body."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    contract: RequestContract
    response: AnswerResponse | None = None
    body_prepared: bool = False
    phase: Literal["awaiting_input", "awaiting_body", "awaiting_destination", "awaiting_delivery"] = "awaiting_destination"
    completed_actions: tuple[Literal["save_text", "slack_notify"], ...] = ()
    save_operation: SaveOperation | None = None
    save_receipt: ActionReceipt | None = None
    slack_delivery: SlackDelivery | None = None


class RuntimeState(BaseModel):
    session_id: str = ""
    user_input: str = ""
    current_turn_id: str = ""
    user_turns: tuple[UserTurnSnapshot, ...] = ()
    retriever: Any | None = None
    upload_files: tuple[UploadFileInfo, ...] = ()
    session_metadata: SessionMetadata = Field(default_factory=SessionMetadata)
    memory_summary: str | None = None
    progress_emitter: Any | None = None
    previous_response: AnswerResponse | None = None
    request_contract: RequestContract | None = None
    pending_action: PendingAction | None = None

    @field_validator("user_turns")
    @classmethod
    def unique_original_turn_ids(cls, value: tuple[UserTurnSnapshot, ...]) -> tuple[UserTurnSnapshot, ...]:
        if len({turn.turn_id for turn in value}) != len(value):
            raise ValueError("duplicate user turn ID in original snapshots")
        return value


class PlannerState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output: PlannerOutput = Field(default_factory=PlannerOutput.fallback)
    status: PlannerStatus = "llm"
    diagnostics: PlannerDiagnostic = Field(default_factory=empty_planner_diagnostic)
    guided_followup: str | None = None


class EvidenceVerdict(BaseModel):
    """A turn-local judgment of one immutable source for one fixed requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    requirement_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    status: Literal["passed", "failed", "refused"]
    probability: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    fingerprint: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_fingerprint: str = Field(min_length=1)
    supported_labels: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_probability(self) -> EvidenceVerdict:
        if (self.status == "refused") != (self.probability is None):
            raise ValueError("only refused evidence has no probability")
        return self


class DecisionCacheEntry(BaseModel):
    """A probability/refusal bound to the complete immutable judgment input."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    fingerprint: str = Field(min_length=1)
    probability: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class PreparedEvidencePacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    requirement_map: dict[str, list[str]] = Field(default_factory=dict)
    failed_requirement_ids: list[str] = Field(default_factory=list)
    fingerprint: str = Field(min_length=1)

    @model_validator(mode="after")
    def binds_selected_ranges(self) -> PreparedEvidencePacket:
        if self.fingerprint != prepared_packet_fingerprint(self.evidence, self.requirement_map):
            raise ValueError("prepared evidence ranges or associations changed")
        return self


class PreparedEvidence(BaseModel):
    """Exact ranges already selected and semantically checked before synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: str
    contract_revision: int
    plan_fingerprint: str
    source_response_hash: str | None = None
    model: str
    prompt_fingerprint: str
    threshold: float = Field(ge=0, le=1)
    normal: PreparedEvidencePacket
    compact: PreparedEvidencePacket | None = None


class RetrievalState(BaseModel):
    hit_log: list[SearchHit] = Field(default_factory=list)
    evidence_verdicts: list[EvidenceVerdict] = Field(default_factory=list)
    decision_cache: list[DecisionCacheEntry] = Field(default_factory=list)
    prepared_evidence: PreparedEvidence | None = None


class ResponseState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result: AnswerResponse = Field(default_factory=AnswerResponse)
    evidence_packet: list[EvidenceRef] = Field(default_factory=list)
    evidence_requirement_map: dict[str, list[str]] = Field(default_factory=dict)
    normal_evidence_missing_requirement_ids: list[str] = Field(default_factory=list)
    synthesis_attempt: int = 0
    kind: Literal["draft", "answer", "acknowledgement", "clarification", "failure"] = "draft"
    problem: ExecutionProblem | None = None
    request_id: str | None = None
    contract_revision: int = 0
    save_operation_binding_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    body_kind: BodyKind | None = None
    evidence_source: AnswerSource | None = None


class DebugState(DebugDiagnostics):
    llm_diagnostics: list[LLMDiagnostic] = Field(default_factory=list)
    planner_errors: list[str] = Field(default_factory=list)
    retrieval_errors: list[str] = Field(default_factory=list)
    synthesis_errors: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    validation_events: list[str] = Field(default_factory=list)
    action_errors: list[str] = Field(default_factory=list)
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    latency_trace: list[dict[str, Any]] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    route_decisions: Annotated[list[RoutingDecision], operator.add]
    runtime: RuntimeState
    planner: PlannerState
    retrieval: RetrievalState
    retry: RetryState
    response: ResponseState
    debug: DebugState
