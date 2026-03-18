from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from .routes import normalize_routes


class _ModelAccessMixin:
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def pop(self, key: str, default: Any = None) -> Any:
        if not hasattr(self, key):
            return default
        value = getattr(self, key)
        setattr(self, key, default)
        return value

    def values(self):
        return self.model_dump(mode="json").values()

    def items(self):
        return self.model_dump(mode="json").items()


RetryReason = Literal[
    "no_evidence",
    "low_score",
    "tool_error",
    "blocked_missing_upload",
    "unsupported_claims",
]
PlannerStatus = Literal["llm", "deterministic", "heuristic_fallback", "fallback_no_routes"]
PlannerOverrideReason = Literal[
    "missing_required_retrieval",
    "missing_required_routes",
    "upload_retriever_missing",
]
LLMCallStage = Literal["summarize", "planner", "synthesis"]
LLMCallPath = Literal[
    "direct",
    "structured",
    "plain_fallback",
    "structured_compact_fallback",
    "plain_summary_attach_fallback",
]

DEFAULT_MAX_RETRIES = 1
RETRYABLE_REASONS: set[RetryReason] = {"no_evidence", "low_score", "tool_error"}


class TokenUsage(_ModelAccessMixin, BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RetryState(_ModelAccessMixin, BaseModel):
    needs_retry: bool = False
    attempt: int = 0
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_reason: RetryReason | None = None
    retrieval_feedback: str = ""
    evidence_start_index: int = 0
    retrieval_error_start_index: int = 0
    retrieval_diagnostic_start_index: int = 0
    score_avg: float | None = None
    failed_routes: list[str] = Field(default_factory=list)
    preserved_evidence: list[dict[str, Any]] = Field(default_factory=list)
    preserved_retrieval_diagnostics: list[dict[str, Any]] = Field(default_factory=list)


class PlannerDiagnostic(_ModelAccessMixin, BaseModel):
    status: str = ""
    reason: str | None = None
    fallback_routes: list[str] = Field(default_factory=list)
    intent_required: bool = False
    required_routes: list[str] = Field(default_factory=list)
    override_applied: bool = False
    override_reason: PlannerOverrideReason | None = None


class RetrievalDiagnostic(_ModelAccessMixin, BaseModel):
    tool: str = ""
    route: str = ""
    status: str = ""
    message: str = ""
    query: str = ""
    attempt: int = 0


class LLMCallMetadata(_ModelAccessMixin, BaseModel):
    stage: LLMCallStage
    attempt: int = 0
    path: LLMCallPath
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)


class DebugPayload(_ModelAccessMixin, BaseModel):
    tool_calls: list[str] = Field(default_factory=list)
    tool_call_count: int = 0
    token_usage: TokenUsage | None = None
    model_name: str | None = None
    models_used: list[str] = Field(default_factory=list)
    llm_calls: list[LLMCallMetadata] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    planner_errors: list[str] = Field(default_factory=list)
    observed_evidence: list[dict[str, Any]] = Field(default_factory=list)
    retry_context: RetryState | None = None
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    planner_diagnostics: PlannerDiagnostic | None = None
    latency_breakdown: dict[str, Any] | None = None


AgentDebugPayload = DebugPayload


def json_safe_deep_copy(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe_deep_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_deep_copy(item) for item in value]
    return str(value)


def empty_planner_diagnostic(*, status: str = "llm") -> PlannerDiagnostic:
    return PlannerDiagnostic(
        status=status,
        reason=None,
        fallback_routes=[],
        intent_required=False,
        required_routes=[],
        override_applied=False,
        override_reason=None,
    )


def coerce_retry_state(value: Any) -> RetryState:
    if isinstance(value, RetryState):
        return value

    retry_state = RetryState()
    if not isinstance(value, dict):
        return retry_state

    attempt = value.get("attempt")
    if isinstance(attempt, int) and attempt >= 0:
        retry_state.attempt = attempt

    max_retries = value.get("max_retries")
    if isinstance(max_retries, int) and max_retries >= 0:
        retry_state.max_retries = max_retries

    retry_reason = value.get("retry_reason")
    if retry_reason in {
        "no_evidence",
        "low_score",
        "tool_error",
        "blocked_missing_upload",
        "unsupported_claims",
    }:
        retry_state.retry_reason = retry_reason

    retry_state.needs_retry = bool(value.get("needs_retry", retry_state.needs_retry))

    retrieval_feedback = value.get("retrieval_feedback")
    if retrieval_feedback is not None:
        retry_state.retrieval_feedback = str(retrieval_feedback).strip()

    evidence_start_index = value.get("evidence_start_index")
    if isinstance(evidence_start_index, int) and evidence_start_index >= 0:
        retry_state.evidence_start_index = evidence_start_index

    retrieval_error_start_index = value.get("retrieval_error_start_index")
    if isinstance(retrieval_error_start_index, int) and retrieval_error_start_index >= 0:
        retry_state.retrieval_error_start_index = retrieval_error_start_index

    retrieval_diagnostic_start_index = value.get("retrieval_diagnostic_start_index")
    if isinstance(retrieval_diagnostic_start_index, int) and retrieval_diagnostic_start_index >= 0:
        retry_state.retrieval_diagnostic_start_index = retrieval_diagnostic_start_index

    score_avg = value.get("score_avg")
    if isinstance(score_avg, (int, float)):
        retry_state.score_avg = float(score_avg)
    elif score_avg is None and "score_avg" in value:
        retry_state.score_avg = None

    failed_routes = value.get("failed_routes")
    if isinstance(failed_routes, list):
        retry_state.failed_routes = normalize_routes(failed_routes)

    preserved_evidence = value.get("preserved_evidence")
    if isinstance(preserved_evidence, list):
        retry_state.preserved_evidence = [
            json_safe_deep_copy(item)
            for item in preserved_evidence
            if isinstance(item, dict)
        ]

    preserved_retrieval_diagnostics = value.get("preserved_retrieval_diagnostics")
    if isinstance(preserved_retrieval_diagnostics, list):
        retry_state.preserved_retrieval_diagnostics = [
            json_safe_deep_copy(item)
            for item in preserved_retrieval_diagnostics
            if isinstance(item, dict)
        ]

    return retry_state


def coerce_planner_diagnostic(value: Any) -> PlannerDiagnostic | None:
    if value is None:
        return None
    if isinstance(value, PlannerDiagnostic):
        return value
    if not isinstance(value, dict):
        return None

    fallback_routes = value.get("fallback_routes")
    required_routes = value.get("required_routes")
    status = value.get("status")
    reason = value.get("reason")
    override_reason = value.get("override_reason")
    if override_reason not in {
        "missing_required_retrieval",
        "missing_required_routes",
        "upload_retriever_missing",
    }:
        override_reason = None

    return PlannerDiagnostic(
        status=str(status) if status is not None else "",
        reason=(str(reason) if reason is not None else None),
        fallback_routes=normalize_routes(fallback_routes) if isinstance(fallback_routes, list) else [],
        intent_required=bool(value.get("intent_required", False)),
        required_routes=normalize_routes(required_routes) if isinstance(required_routes, list) else [],
        override_applied=bool(value.get("override_applied", False)),
        override_reason=override_reason,
    )


def coerce_retrieval_diagnostic(value: Any) -> RetrievalDiagnostic | None:
    if isinstance(value, RetrievalDiagnostic):
        return value
    if not isinstance(value, dict):
        return None
    try:
        attempt = int(value.get("attempt", 0) or 0)
    except (TypeError, ValueError):
        attempt = 0
    return RetrievalDiagnostic(
        tool=str(value.get("tool") or "").strip(),
        route=str(value.get("route") or "").strip(),
        status=str(value.get("status") or "").strip(),
        message=str(value.get("message") or ""),
        query=str(value.get("query") or ""),
        attempt=max(0, attempt),
    )


def coerce_retrieval_diagnostics(value: Any) -> list[RetrievalDiagnostic]:
    if not isinstance(value, list):
        return []
    diagnostics: list[RetrievalDiagnostic] = []
    for item in value:
        diagnostic = coerce_retrieval_diagnostic(item)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return diagnostics


def coerce_llm_calls(value: Any) -> list[LLMCallMetadata]:
    if not isinstance(value, list):
        return []
    calls: list[LLMCallMetadata] = []
    for item in value:
        if isinstance(item, LLMCallMetadata):
            calls.append(item)
            continue
        if not isinstance(item, dict):
            continue

        stage = str(item.get("stage") or "").strip()
        if stage not in {"summarize", "planner", "synthesis"}:
            continue

        path = str(item.get("path") or "").strip()
        if path not in {
            "direct",
            "structured",
            "plain_fallback",
            "structured_compact_fallback",
            "plain_summary_attach_fallback",
        }:
            continue

        try:
            attempt = int(item.get("attempt", 0) or 0)
        except (TypeError, ValueError):
            attempt = 0

        response_metadata = item.get("response_metadata")
        usage_metadata = item.get("usage_metadata")
        calls.append(
            LLMCallMetadata(
                stage=stage,
                attempt=max(0, attempt),
                path=path,
                response_metadata=dict(response_metadata) if isinstance(response_metadata, dict) else {},
                usage_metadata=dict(usage_metadata) if isinstance(usage_metadata, dict) else {},
            )
        )
    return calls


def coerce_token_usage(value: Any) -> TokenUsage | None:
    if isinstance(value, TokenUsage):
        return value
    if not isinstance(value, dict):
        return None
    try:
        return TokenUsage(
            prompt_tokens=int(value.get("prompt_tokens", 0) or 0),
            completion_tokens=int(value.get("completion_tokens", 0) or 0),
            total_tokens=int(value.get("total_tokens", 0) or 0),
        )
    except (TypeError, ValueError):
        return None


def coerce_debug_payload(value: Any) -> DebugPayload:
    if isinstance(value, DebugPayload):
        return value
    if not isinstance(value, dict):
        return DebugPayload()

    observed_evidence = [
        dict(item)
        for item in value.get("observed_evidence", [])
        if isinstance(item, dict)
    ] if isinstance(value.get("observed_evidence"), list) else []
    return DebugPayload(
        tool_calls=[str(item) for item in value.get("tool_calls", []) if str(item).strip()]
        if isinstance(value.get("tool_calls"), list)
        else [],
        tool_call_count=int(value.get("tool_call_count", 0) or 0),
        token_usage=coerce_token_usage(value.get("token_usage")),
        model_name=str(value.get("model_name")) if value.get("model_name") else None,
        models_used=[str(item) for item in value.get("models_used", []) if str(item).strip()]
        if isinstance(value.get("models_used"), list)
        else [],
        llm_calls=coerce_llm_calls(value.get("llm_calls")),
        errors=[str(item) for item in value.get("errors", []) if str(item).strip()]
        if isinstance(value.get("errors"), list)
        else [],
        planner_errors=[str(item) for item in value.get("planner_errors", []) if str(item).strip()]
        if isinstance(value.get("planner_errors"), list)
        else [],
        observed_evidence=observed_evidence,
        retry_context=coerce_retry_state(value.get("retry_context")) if value.get("retry_context") else None,
        retrieval_diagnostics=coerce_retrieval_diagnostics(value.get("retrieval_diagnostics")),
        planner_diagnostics=coerce_planner_diagnostic(value.get("planner_diagnostics")),
        latency_breakdown=dict(value.get("latency_breakdown"))
        if isinstance(value.get("latency_breakdown"), dict)
        else None,
    )


def build_llm_call_metadata(
    *,
    stage: LLMCallStage,
    attempt: int,
    path: LLMCallPath,
    message: AIMessage,
) -> LLMCallMetadata:
    response_metadata = getattr(message, "response_metadata", None)
    usage_metadata = getattr(message, "usage_metadata", None)

    safe_response_metadata = (
        json_safe_deep_copy(response_metadata) if isinstance(response_metadata, dict) else {}
    )
    safe_usage_metadata = (
        json_safe_deep_copy(usage_metadata) if isinstance(usage_metadata, dict) else {}
    )

    return LLMCallMetadata(
        stage=stage,
        attempt=max(0, int(attempt)),
        path=path,
        response_metadata=safe_response_metadata,
        usage_metadata=safe_usage_metadata,
    )
