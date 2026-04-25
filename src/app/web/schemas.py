from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.answer_schema import AgentResponsePayloadModel
from src.core.contracts.debug import ActionResults, ErrorCode, LLMCallMetadata, PlannerDiagnostic, RetryState, RetrievalDiagnostic, TokenUsage
from src.core.evidence import EvidenceItem
from src.core.latency import LatencyBreakdownModel, StageName

AgentResponsePayload = AgentResponsePayloadModel
AgentTokenUsage = TokenUsage
AgentRetryContext = RetryState
PlannerMode = Literal["auto", "force_llm"]


class AgentDebugInfo(BaseModel):
    schema_version: int
    observability_status: Literal["ok", "degraded", "failed"]
    missing_required_debug_fields: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    tool_call_count: int = 0
    latency_ms_server: int | None = None
    latency_breakdown: LatencyBreakdownModel | None = None
    token_usage: AgentTokenUsage | None = None
    model_name: str | None = None
    models_used: list[str] = Field(default_factory=list)
    llm_calls: list[LLMCallMetadata] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    error_codes: list[ErrorCode] = Field(default_factory=list)
    validation_events: list[str] = Field(default_factory=list)
    edge_decisions: list[dict[str, Any]] = Field(default_factory=list)
    planner_errors: list[str] = Field(default_factory=list)
    observed_evidence: list[EvidenceItem] = Field(default_factory=list)
    retry_context: AgentRetryContext | None = None
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    planner_diagnostics: PlannerDiagnostic | None = None
    action_results: ActionResults | None = None


class AgentRequest(BaseModel):
    query: str
    session_id: str
    slack_user_id: str | None = None
    slack_email: str | None = None
    slack_channel_id: str | None = None
    upload_file_path: str | None = None
    upload_file_paths: list[str] | None = None
    reset_slack_destination: bool = False
    planner_mode: PlannerMode = "auto"
    eval_faults: dict[str, str] = Field(default_factory=dict)
    include_debug: bool = False


class AgentResponse(BaseModel):
    response: AgentResponsePayload
    trace: str
    file_path: str | None = None
    debug: AgentDebugInfo | None = None


AgentStreamEventName = Literal[
    "request_started",
    "stage_started",
    "stage_completed",
    "heartbeat",
    "progress_snapshot",
    "final_response",
    "error",
    "done",
]
AgentStreamStageName = StageName


class AgentStreamEvent(BaseModel):
    event: AgentStreamEventName
    data: dict[str, Any] = Field(default_factory=dict)
