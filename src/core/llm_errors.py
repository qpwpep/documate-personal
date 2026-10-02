"""Stable failures shared by LLM callers and the user-facing turn boundary."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


FailureCode = Literal[
    "provider_schema_invalid", "provider_configuration", "provider_unavailable",
    "provider_rate_limited", "model_output_invalid", "model_output_incomplete",
    "model_refusal", "internal_error", "evidence_insufficient", "upload_revision_conflict",
    "call_budget_exhausted",
]
NextAction = Literal["none", "retry_later", "fix_configuration", "supply_information"]


class ExecutionProblem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: FailureCode
    stage: str
    message: str
    next_action: NextAction
    retry_after_seconds: float | None = Field(default=None, ge=0)


_PUBLIC_MESSAGES: dict[str, tuple[str, NextAction]] = {
    "provider_schema_invalid": ("서비스의 요청 처리 설정에 오류가 있어 처리를 완료하지 못했습니다. 질문을 수정할 필요는 없습니다.", "fix_configuration"),
    "provider_configuration": ("AI 서비스 연결 설정을 확인해야 해 처리를 완료하지 못했습니다. 질문을 수정할 필요는 없습니다.", "fix_configuration"),
    "provider_unavailable": ("AI 서비스 연결이 일시적으로 원활하지 않아 처리를 완료하지 못했습니다. 잠시 후 같은 요청으로 다시 시도할 수 있습니다.", "retry_later"),
    "provider_rate_limited": ("AI 서비스의 요청이 일시적으로 몰려 처리를 완료하지 못했습니다. 잠시 후 같은 요청으로 다시 시도할 수 있습니다.", "retry_later"),
    "model_output_invalid": ("AI가 생성한 결과를 검증하지 못해 처리를 완료하지 못했습니다. 질문을 수정할 필요는 없습니다.", "retry_later"),
    "model_output_incomplete": ("AI의 출력이 끝까지 생성되지 않아 처리를 완료하지 못했습니다. 잠시 후 같은 요청으로 다시 시도할 수 있습니다.", "retry_later"),
    "model_refusal": ("AI 서비스가 이 요청에 대한 답변 생성을 거절했습니다.", "none"),
    "call_budget_exhausted": ("이번 요청의 AI 처리 시간 또는 호출 한도에 도달했습니다. 질문을 수정하지 않고 같은 요청으로 다시 시도할 수 있습니다.", "retry_later"),
    "internal_error": ("서비스 내부 오류로 처리를 완료하지 못했습니다. 질문을 수정할 필요는 없습니다.", "none"),
    "evidence_insufficient": ("요청한 답변을 뒷받침할 근거를 충분히 확인하지 못했습니다.", "supply_information"),
    "upload_revision_conflict": ("첨부 목록이 변경되었습니다. 현재 첨부 목록을 확인한 뒤 다시 요청해 주세요.", "none"),
}


def make_problem(code: FailureCode, stage: str, *, retry_after_seconds: float | None = None) -> ExecutionProblem:
    message, action = _PUBLIC_MESSAGES[code]
    return ExecutionProblem(code=code, stage=stage, message=message, next_action=action,
                            retry_after_seconds=retry_after_seconds)


class LLMDiagnostic(BaseModel):
    """Safe metadata only: never store exception text, prompts, or generated values."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: FailureCode
    stage: str
    model: str | None = None
    endpoint: str | None = None
    schema_name: str | None = None
    schema_hash: str | None = None
    schema_compiler_version: str | None = None
    openai_sdk_version: str | None = None
    langchain_openai_version: str | None = None
    provider_status: int | None = None
    provider_code: str | None = None
    provider_type: str | None = None
    provider_param: str | None = None
    provider_request_id: str | None = None
    attempt: int = 0
    exception_type: str | None = None
    transport_failure: Literal["timeout", "connection"] | None = None
    budget_reason: Literal["calls", "deadline"] | None = None
    validation_paths: list[str] = Field(default_factory=list)
    recovery: str = "stop"


class LLMCallError(RuntimeError):
    def __init__(self, problem: ExecutionProblem, diagnostic: LLMDiagnostic | None = None):
        super().__init__(problem.message)
        self.problem = problem
        self.diagnostic = diagnostic or LLMDiagnostic(code=problem.code, stage=problem.stage)
