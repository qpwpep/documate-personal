"""One interpretation of provider token observations, independent of consumers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


TokenCount = Annotated[int, Field(strict=True, ge=0)]
LLMCallStage = Literal[
    "summarize", "planner", "synthesis", "pre_synthesis_validation", "post_synthesis_validation",
]
LLMCallPath = Literal["direct", "structured", "structured_compact_fallback", "decisions"]
UsageField = Literal[
    "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
    "cache_write_input_tokens", "reasoning_output_tokens", "metadata",
]


class UsageIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: Literal["invalid_value", "conflict", "total_mismatch", "detail_exceeds_total"]
    field: UsageField
    source: str


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    cached_input_tokens: TokenCount | None = None
    cache_write_input_tokens: TokenCount | None = None
    reasoning_output_tokens: TokenCount | None = None
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

    @property
    def uncached_input_tokens(self) -> int | None:
        if self.input_tokens is None or self.cached_input_tokens is None or self.cache_write_input_tokens is None:
            return None
        remaining = self.input_tokens - self.cached_input_tokens - self.cache_write_input_tokens
        return remaining if remaining >= 0 else None


class LLMCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: LLMCallStage
    attempt: TokenCount = 0
    path: LLMCallPath
    model_name: Annotated[str, Field(strict=True, min_length=1)] | None = None
    endpoint: str | None = None
    status: Literal["unknown", "started", "completed", "failed"] = "unknown"
    latency_ms: TokenCount | None = None
    failure_type: str | None = None
    usage: TokenUsage
    # Retained for diagnostics; no consumer may interpret these again.
    response_metadata: dict[str, object] = Field(default_factory=dict)
    usage_metadata: dict[str, object] = Field(default_factory=dict)


def normalize_token_usage(usage_metadata: object, response_metadata: object) -> TokenUsage:
    """Select each dimension independently, without coercing absent or invalid data.

    Original provider input/output counts are authoritative when retained: SDKs
    can synthesize zero for an absent count. Otherwise use the SDK observation.
    Nested SDK details precede response details; a valid zero is authoritative.
    Reported totals never fill missing dimensions.
    """
    issues: list[UsageIssue] = []
    candidates: list[tuple[str, dict[object, object]]] = []

    def add_candidate(source: str, value: object) -> None:
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

    def select(field: UsageField, paths: tuple[tuple[str, ...], ...], *, provider_counts: bool = False) -> int | None:
        selected: int | None = None
        provider = [candidate for candidate in candidates if candidate[0] == "response_metadata.token_usage"]
        observations = provider if provider_counts and provider else candidates
        for source, candidate in observations:
            for path in paths:
                value: object = candidate
                for key in path:
                    if value is None:
                        break
                    if not isinstance(value, dict):
                        issues.append(UsageIssue(code="invalid_value", field=field, source=source + "." + ".".join(path)))
                        value = None
                        break
                    value = value.get(key)
                if value is None:
                    continue
                location = source + "." + ".".join(path)
                if type(value) is not int or value < 0:
                    issues.append(UsageIssue(code="invalid_value", field=field, source=location))
                elif selected is None:
                    selected = value
                elif selected != value:
                    issues.append(UsageIssue(code="conflict", field=field, source=location))
        return selected

    input_tokens = select("input_tokens", (("input_tokens",), ("prompt_tokens",)), provider_counts=True)
    output_tokens = select("output_tokens", (("output_tokens",), ("completion_tokens",)), provider_counts=True)
    cached_input_tokens = select("cached_input_tokens", (
        ("input_token_details", "cache_read"), ("input_tokens_details", "cached_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
    ))
    cache_write_input_tokens = select("cache_write_input_tokens", (
        ("input_token_details", "cache_creation"), ("input_tokens_details", "cache_write_tokens"),
        ("prompt_tokens_details", "cache_write_tokens"),
    ))
    reasoning_output_tokens = select("reasoning_output_tokens", (
        ("output_token_details", "reasoning"), ("output_tokens_details", "reasoning_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
    ))
    total = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
    for source, candidate in candidates:
        reported = candidate.get("total_tokens")
        if reported is None:
            continue
        if type(reported) is not int or reported < 0:
            issues.append(UsageIssue(code="invalid_value", field="total_tokens", source=f"{source}.total_tokens"))
        elif total is not None and total != reported:
            issues.append(UsageIssue(code="total_mismatch", field="total_tokens", source=f"{source}.total_tokens"))
    for field, detail, parent in (
        ("cached_input_tokens", cached_input_tokens, input_tokens),
        ("cache_write_input_tokens", cache_write_input_tokens, input_tokens),
        ("reasoning_output_tokens", reasoning_output_tokens, output_tokens),
    ):
        if detail is not None and parent is not None and detail > parent:
            issues.append(UsageIssue(code="detail_exceeds_total", field=field, source=field))
    if (input_tokens is not None and cached_input_tokens is not None and cache_write_input_tokens is not None
            and cached_input_tokens + cache_write_input_tokens > input_tokens):
        issues.append(UsageIssue(code="detail_exceeds_total", field="input_tokens", source="cache_read+cache_write"))
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens,
                      cached_input_tokens=cached_input_tokens, cache_write_input_tokens=cache_write_input_tokens,
                      reasoning_output_tokens=reasoning_output_tokens, issues=issues)
