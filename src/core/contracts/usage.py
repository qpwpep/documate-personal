"""One interpretation of provider token observations, independent of consumers."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TokenCount = Annotated[int, Field(strict=True, ge=0)]
LLMCallStage = Literal["summarize", "planner", "synthesis"]
LLMCallPath = Literal["direct", "structured", "structured_compact_fallback"]


class UsageIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: Literal["invalid_value", "conflict", "total_mismatch"]
    field: Literal["input_tokens", "output_tokens", "total_tokens", "metadata"]
    source: str


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    issues: list[UsageIssue] = Field(default_factory=list)

    @property
    def observation(self) -> Literal["complete", "partial", "none"]:
        observed = sum(value is not None for value in (self.input_tokens, self.output_tokens))
        return ("none", "partial", "complete")[observed]

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


class LLMCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: LLMCallStage
    attempt: TokenCount = 0
    path: LLMCallPath
    model_name: Annotated[str, Field(strict=True, min_length=1)] | None = None
    usage: TokenUsage
    # Retained for diagnostics; no consumer may interpret these again.
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)


def normalize_token_usage(usage_metadata: Any, response_metadata: Any) -> TokenUsage:
    """Select each dimension independently, without coercing absent or invalid data.

    SDK usage precedes response token_usage, input/output precede their aliases.
    A valid zero is authoritative. Lower-priority disagreements are diagnostic.
    Reported totals never fill missing dimensions.
    """
    issues: list[UsageIssue] = []
    candidates: list[tuple[str, dict[str, Any]]] = []

    def add_candidate(source: str, value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            issues.append(UsageIssue(code="invalid_value", field="metadata", source=source))
            return
        candidates.append((source, value))

    add_candidate("usage_metadata", usage_metadata)
    if response_metadata is not None:
        if isinstance(response_metadata, dict):
            add_candidate("response_metadata.token_usage", response_metadata.get("token_usage"))
        else:
            issues.append(UsageIssue(code="invalid_value", field="metadata", source="response_metadata"))

    def select(field: Literal["input_tokens", "output_tokens"], alias: str) -> int | None:
        selected: int | None = None
        for source, candidate in candidates:
            for key in (field, alias):
                value = candidate.get(key)
                if value is None:
                    continue
                location = f"{source}.{key}"
                if type(value) is not int or value < 0:
                    issues.append(UsageIssue(code="invalid_value", field=field, source=location))
                elif selected is None:
                    selected = value
                elif selected != value:
                    issues.append(UsageIssue(code="conflict", field=field, source=location))
        return selected

    input_tokens = select("input_tokens", "prompt_tokens")
    output_tokens = select("output_tokens", "completion_tokens")
    total = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
    for source, candidate in candidates:
        reported = candidate.get("total_tokens")
        if reported is None:
            continue
        if type(reported) is not int or reported < 0:
            issues.append(UsageIssue(code="invalid_value", field="total_tokens", source=f"{source}.total_tokens"))
        elif total is not None and total != reported:
            issues.append(UsageIssue(code="total_mismatch", field="total_tokens", source=f"{source}.total_tokens"))
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, issues=issues)
