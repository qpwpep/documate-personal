from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.core.answer_schema import AgentResponsePayloadModel, SynthesisOutput, build_empty_response_payload
from src.core.planner_schema import PlannerOutput
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


class RuntimeState(BaseModel):
    user_input: str = ""
    retriever: Any | None = None
    session_metadata: SessionMetadata = Field(default_factory=SessionMetadata)
    memory_summary: str | None = None
    progress_emitter: Any | None = None
    planner_mode: Literal["auto", "force_llm"] = "auto"
    eval_faults: dict[str, str] = Field(default_factory=dict)


class PlannerState(BaseModel):
    output: PlannerOutput = Field(default_factory=PlannerOutput.fallback)
    status: PlannerStatus = "llm"
    diagnostics: PlannerDiagnostic = Field(default_factory=empty_planner_diagnostic)
    guided_followup: str | None = None


class RetrievalState(BaseModel):
    evidence_log: list[dict[str, Any]] = Field(default_factory=list)


class ResponseState(BaseModel):
    final_answer: str = ""
    payload: AgentResponsePayloadModel = Field(default_factory=build_empty_response_payload)
    synthesis_output: SynthesisOutput = Field(default_factory=SynthesisOutput)
    synthesis_attempt: int = 0


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
