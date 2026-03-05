from __future__ import annotations

from pydantic import BaseModel, Field

from ..evidence import EvidenceItem


class AgentResponsePayload(BaseModel):
    answer: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)


class AgentTokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentDebugInfo(BaseModel):
    tool_calls: list[str] = Field(default_factory=list)
    tool_call_count: int = 0
    latency_ms_server: int | None = None
    token_usage: AgentTokenUsage | None = None
    model_name: str | None = None
    errors: list[str] = Field(default_factory=list)
    observed_evidence: list[EvidenceItem] = Field(default_factory=list)


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
