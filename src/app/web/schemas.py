from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.answer_schema import AnswerResponse
from src.core.conversation_memory import (
    DEFAULT_QUERY_MAX_CHARS,
    validate_query_text,
)
from src.core.contracts.debug import ActionResults, ErrorCode, LLMCallMetadata, ModelUsageStatus, PlannerDiagnostic, RetryState, RetrievalDiagnostic, TokenUsage
from src.core.evidence import SearchHit
from src.core.uploads import UploadContext, UploadManifest, validate_session_id
from src.core.latency import LatencyBreakdownModel, StageName

AgentTokenUsage = TokenUsage
AgentRetryContext = RetryState


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
    model_usage_status: ModelUsageStatus = "missing_debug"
    llm_calls: list[LLMCallMetadata] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    error_codes: list[ErrorCode] = Field(default_factory=list)
    validation_events: list[str] = Field(default_factory=list)
    edge_decisions: list[dict[str, Any]] = Field(default_factory=list)
    planner_errors: list[str] = Field(default_factory=list)
    observed_hits: list[SearchHit] = Field(default_factory=list)
    retry_context: AgentRetryContext | None = None
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    planner_diagnostics: PlannerDiagnostic | None = None
    action_results: ActionResults | None = None


class AgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=DEFAULT_QUERY_MAX_CHARS)
    session_id: str
    slack_user_id: str | None = None
    slack_email: str | None = None
    slack_channel_id: str | None = None
    upload_file_path: str | None = None
    uploads: UploadContext | None = None
    include_debug: bool = False

    @field_validator("session_id")
    @classmethod
    def _validate_session(cls, value: str) -> str:
        return validate_session_id(value)

    @model_validator(mode="after")
    def _validate_upload_contract(self) -> AgentRequest:
        if "uploads" in self.model_fields_set:
            if self.uploads is None:
                raise ValueError("uploads must specify an epoch and revision")
            if "upload_file_path" in self.model_fields_set:
                raise ValueError("uploads and upload_file_path cannot be combined")
        return self

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        return validate_query_text(value)


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: AnswerResponse
    trace: str
    debug: AgentDebugInfo | None = None
    upload_manifest: UploadManifest | None = Field(
        default=None,
        description="Attachment state captured under the session lock after this request; absent in older responses.",
    )


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


def _event_data_schema(required: list[str], **properties: Any) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


_STREAM_STAGE_SCHEMA = {"type": "string", "enum": list(get_args(StageName))}
_STRING_SCHEMA = {"type": "string"}
_INTEGER_SCHEMA = {"type": "integer"}

# Each entry describes the JSON value in the named SSE frame's data field.
# final_response refers to the same runtime model used by response assembly.
AGENT_STREAM_EVENT_SCHEMAS = {
    "request_started": _event_data_schema(
        ["request_id", "session_id"], request_id=_STRING_SCHEMA, session_id=_STRING_SCHEMA,
    ),
    "stage_started": _event_data_schema(
        ["stage", "attempt"], stage=_STREAM_STAGE_SCHEMA, attempt=_INTEGER_SCHEMA,
    ),
    "stage_completed": _event_data_schema(
        ["stage", "attempt"], stage=_STREAM_STAGE_SCHEMA, attempt=_INTEGER_SCHEMA,
        latency_ms=_INTEGER_SCHEMA, status=_STRING_SCHEMA,
    ),
    "heartbeat": _event_data_schema(
        ["stage", "attempt", "elapsed_ms"], stage=_STREAM_STAGE_SCHEMA,
        attempt=_INTEGER_SCHEMA, elapsed_ms=_INTEGER_SCHEMA,
    ),
    "progress_snapshot": _event_data_schema(
        ["stage", "summary"], stage=_STREAM_STAGE_SCHEMA, summary=_STRING_SCHEMA,
    ),
    "final_response": {"$ref": "#/components/schemas/AgentResponse"},
    "error": _event_data_schema(["message"], message=_STRING_SCHEMA, stage=_STREAM_STAGE_SCHEMA),
    "done": {"type": "object", "maxProperties": 0},
}
