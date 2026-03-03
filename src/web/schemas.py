from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EvidenceKind = Literal["official", "local"]


class EvidenceItem(BaseModel):
    kind: EvidenceKind
    source: str
    title: str | None = None
    snippet: str | None = None
    tool: str
    source_id: str


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


# 입력 모델
class AgentRequest(BaseModel):
    query: str
    session_id: str  # 클라이언트가 생성하여 전송해야 하는 고유 세션 ID (멀티턴 처리 시 사용)
    slack_user_id: str | None = None  # DM 대상 Uxxxxx
    slack_email: str | None = None  # DM 대상 이메일
    slack_channel_id: str | None = None  # 채널/그룹/DM 채널 ID (C/G/Dxxxxx)
    upload_file_path: str | None = None  # 업로드한 파일 경로
    include_debug: bool = False  # 벤치마크/평가용 디버그 정보 포함 여부


# 출력 모델
class AgentResponse(BaseModel):
    response: AgentResponsePayload
    trace: str  # 출력 확인용
    file_path: str | None = None  # 파일이 생성된 경우, 해당 파일 경로
    debug: AgentDebugInfo | None = None
