from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..evidence import EvidenceItem


class AgentResponsePayload(BaseModel):
    answer: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)


class AgentTokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentRetryContext(BaseModel):
    attempt: int = 0
    max_retries: int = 1
    retry_reason: Literal["no_evidence", "low_score", "tool_error", "blocked_missing_upload"] | None = None
    retrieval_feedback: str | None = None
    evidence_start_index: int | None = None
    retrieval_error_start_index: int | None = None
    score_avg: float | None = None


class RetrievalDiagnostic(BaseModel):
    tool: str = ""
    route: str = ""
    status: str = ""
    message: str = ""
    query: str = ""
    attempt: int = 0


class PlannerDiagnostic(BaseModel):
    status: str = ""
    reason: str | None = None
    fallback_routes: list[str] = Field(default_factory=list)


class AgentDebugInfo(BaseModel):
    tool_calls: list[str] = Field(default_factory=list)
    tool_call_count: int = 0
    latency_ms_server: int | None = None
    token_usage: AgentTokenUsage | None = None
    model_name: str | None = None
    errors: list[str] = Field(default_factory=list)
    observed_evidence: list[EvidenceItem] = Field(default_factory=list)
    retry_context: AgentRetryContext | None = None
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    planner_diagnostics: PlannerDiagnostic | None = None


class AgentRequest(BaseModel):
    query: str
    session_id: str
    slack_user_id: str | None = None
    slack_email: str | None = None
    slack_channel_id: str | None = None
    upload_file_path: str | None = None
    include_debug: bool = False


class AgentResponse(BaseModel):
    response: AgentResponsePayload
    trace: str
    file_path: str | None = None
    debug: AgentDebugInfo | None = None
