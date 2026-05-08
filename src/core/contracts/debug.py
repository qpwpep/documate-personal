from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

ErrorCode = Literal[
    "PLANNER_SCHEMA_INVALID",
    "PLANNER_TIMEOUT",
    "RETRIEVAL_DOCS_TIMEOUT",
    "RETRIEVAL_DOCS_FAILED",
    "RAG_INDEX_MISSING",
    "LOCAL_RAG_FAILED",
    "UPLOAD_RETRIEVER_BUILD_FAILED",
    "LLM_STRUCTURED_EMPTY",
    "SYNTHESIS_TIMEOUT",
    "VALIDATION_UNSUPPORTED_CLAIMS",
    "HYBRID_SECTION_REPEATED",
    "HYBRID_UPLOAD_SETTING_MISSING",
    "HYBRID_COMPARISON_WEAK",
    "DEBUG_NORMALIZATION_FAILED",
    "SLACK_AUTH_FAILED",
    "SLACK_DESTINATION_MISSING",
    "UPLOAD_PATH_INVALID",
]
RetryReason = Literal[
    "no_evidence",
    "low_score",
    "tool_error",
    "blocked_missing_upload",
    "unsupported_claims",
    "missing",
    "missing_route_coverage",
    "missing_sections",
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
    "korean_template_summary_fallback",
]

DEFAULT_MAX_RETRIES = 1
RETRYABLE_REASONS: set[RetryReason] = {
    "no_evidence",
    "low_score",
    "tool_error",
    "missing",
}
DEBUG_SCHEMA_VERSION = 4
DebugObservabilityStatus = Literal["ok", "degraded", "failed"]
ModelUsageStatus = Literal["llm_used", "deterministic", "missing_debug"]
DEBUG_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "observability_status",
    "missing_required_debug_fields",
    "tool_calls",
    "tool_call_count",
    "token_usage",
    "model_name",
    "models_used",
    "model_usage_status",
    "llm_calls",
    "errors",
    "planner_errors",
    "observed_evidence",
    "retry_context",
    "retrieval_diagnostics",
    "planner_diagnostics",
    "latency_breakdown",
)
DEBUG_CRITICAL_FIELDS: tuple[str, ...] = (
    "schema_version",
    "observability_status",
    "missing_required_debug_fields",
    "tool_calls",
    "tool_call_count",
    "observed_evidence",
    "retrieval_diagnostics",
    "latency_breakdown",
)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RetryState(BaseModel):
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
    preserved_retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    retry_scope: Literal["refresh_routes", "reuse_evidence_resynthesize"] = "refresh_routes"


class PlannerDiagnostic(BaseModel):
    status: str = ""
    reason: str | None = None
    fallback_routes: list[str] = Field(default_factory=list)
    intent_required: bool = False
    required_routes: list[str] = Field(default_factory=list)
    override_applied: bool = False
    override_reason: PlannerOverrideReason | None = None
    planner_warnings: list[str] = Field(default_factory=list)


class RetrievalDiagnostic(BaseModel):
    tool: str = ""
    route: str = ""
    status: str = ""
    message: str = ""
    error_code: ErrorCode | None = None
    query: str = ""
    attempt: int = 0
    evidence_count: int = 0
    result_count: int = 0
    provider_result_count: int = 0
    filtered_invalid_url_count: int = 0
    filtered_path_prefix_count: int = 0
    filtered_cross_domain_count: int = 0
    filtered_http_error_count: int = 0
    filtered_redirect_policy_count: int = 0
    filtered_url_request_failed_count: int = 0
    filtered_identifier_mismatch_count: int = 0
    validated_url_count: int = 0
    final_evidence_count: int = 0
    metric: str = ""
    score_direction: Literal["higher_is_better", "lower_is_better", ""] = ""
    normalized_score: float | None = None
    raw_score: float | None = None
    provider_ms: int = 0
    url_validation_ms: int = 0
    post_filter_ms: int = 0
    include_raw_content_requested: bool = False
    warnings: list[str] = Field(default_factory=list)


class LLMCallMetadata(BaseModel):
    stage: LLMCallStage
    attempt: int = 0
    path: LLMCallPath
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)


class SlackActionResult(BaseModel):
    status: str = ""
    channel_id: str | None = None
    target_type: str | None = None
    error: str | None = None
    reason: str | None = None
    error_code: ErrorCode | None = None


class SaveTextActionResult(BaseModel):
    status: str = ""
    file_path: str | None = None
    bytes: int = 0
    error: str | None = None
    message: str | None = None
    error_code: ErrorCode | None = None


class ActionResults(BaseModel):
    slack_notify: SlackActionResult | None = None
    save_text: SaveTextActionResult | None = None


class DebugPayload(BaseModel):
    schema_version: int = DEBUG_SCHEMA_VERSION
    observability_status: DebugObservabilityStatus = "ok"
    missing_required_debug_fields: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    tool_call_count: int = 0
    token_usage: TokenUsage | None = None
    model_name: str | None = None
    models_used: list[str] = Field(default_factory=list)
    model_usage_status: ModelUsageStatus = "deterministic"
    llm_calls: list[LLMCallMetadata] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    error_codes: list[ErrorCode] = Field(default_factory=list)
    validation_events: list[str] = Field(default_factory=list)
    edge_decisions: list[dict[str, Any]] = Field(default_factory=list)
    planner_errors: list[str] = Field(default_factory=list)
    observed_evidence: list[dict[str, Any]] = Field(default_factory=list)
    retry_context: RetryState | None = None
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    planner_diagnostics: PlannerDiagnostic | None = None
    latency_breakdown: dict[str, Any] | None = None
    action_results: ActionResults | None = None


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
        planner_warnings=[],
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
