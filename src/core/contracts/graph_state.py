from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypedDict

from src.core.answer_schema import AnswerResponse
from src.core.evidence import EvidenceRef
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import RequestContract, UserTurnSnapshot
from src.core.contracts.debug import DebugPayload, LLMCallMetadata, PlannerDiagnostic, PlannerStatus, RetryState, RetrievalDiagnostic, empty_planner_diagnostic


class SlackDestination(BaseModel):
    channel_id: str | None = None
    user_id: str | None = None
    email: str | None = None

    def has_destination(self) -> bool:
        return any(
            value is not None and str(value).strip()
            for value in (self.channel_id, self.user_id, self.email)
        )


class SessionMetadata(BaseModel):
    slack_destination: SlackDestination | None = None


class PendingAction(BaseModel):
    """An unfinished request owns its requirements and any checked body."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    contract: RequestContract
    response: AnswerResponse | None = None
    body_prepared: bool = False
    phase: Literal["awaiting_input", "awaiting_body", "awaiting_destination", "awaiting_delivery"] = "awaiting_destination"
    completed_actions: tuple[Literal["save_text", "slack_notify"], ...] = ()


class RuntimeState(BaseModel):
    user_input: str = ""
    current_turn_id: str = ""
    user_turns: tuple[UserTurnSnapshot, ...] = ()
    retriever: Any | None = None
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
    output: PlannerOutput = Field(default_factory=PlannerOutput.fallback)
    status: PlannerStatus = "llm"
    diagnostics: PlannerDiagnostic = Field(default_factory=empty_planner_diagnostic)
    guided_followup: str | None = None


class RetrievalState(BaseModel):
    hit_log: list[dict[str, Any]] = Field(default_factory=list)


class ResponseState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result: AnswerResponse = Field(default_factory=AnswerResponse)
    evidence_packet: list[EvidenceRef] = Field(default_factory=list)
    evidence_requirement_map: dict[str, list[str]] = Field(default_factory=dict)
    synthesis_attempt: int = 0
    kind: Literal["draft", "answer", "clarification", "failure"] = "draft"
    request_id: str | None = None
    contract_revision: int = 0


class DebugState(DebugPayload):
    planner_errors: list[str] = Field(default_factory=list)
    retrieval_errors: list[str] = Field(default_factory=list)
    synthesis_errors: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    validation_events: list[str] = Field(default_factory=list)
    edge_decisions: list[dict[str, Any]] = Field(default_factory=list)
    action_errors: list[str] = Field(default_factory=list)
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    llm_calls: list[LLMCallMetadata] = Field(default_factory=list)
    latency_trace: list[dict[str, Any]] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    runtime: RuntimeState
    planner: PlannerState
    retrieval: RetrievalState
    retry: RetryState
    response: ResponseState
    debug: DebugState
