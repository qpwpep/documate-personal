"""Request-scoped facts about entry into registered tool implementations."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EXECUTABLE_TOOL_NAMES = frozenset({"tavily_search", "upload_search", "save_text", "slack_notify"})


class ToolExecutionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1, strict=True)
    invocation_id: str = Field(min_length=1, strict=True)
    tool_name: str = Field(min_length=1, strict=True)
    phase: Literal["started", "succeeded", "failed", "blocked", "reused"]
    reason_code: str | None = None
    origin_invocation_id: str | None = None

    @field_validator("tool_name")
    @classmethod
    def canonical_tool_name(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("execution tool name must be nonblank and have no surrounding whitespace")
        return value


class ToolExecutionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    request_id: str = Field(min_length=1, strict=True)
    status: Literal["complete", "incomplete", "unavailable"]
    events: list[ToolExecutionEvent]

    @field_validator("schema_version", mode="before")
    @classmethod
    def explicit_integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("execution schema version must be an integer")
        return value

    @model_validator(mode="after")
    def consistent_execution_history(self) -> "ToolExecutionEvidence":
        opened: dict[str, str] = {}
        seen: set[str] = set()
        for sequence, event in enumerate(self.events, 1):
            if event.sequence != sequence:
                raise ValueError("execution event sequence must be contiguous and ordered")
            if event.phase in {"started", "blocked", "reused"}:
                if event.invocation_id in seen:
                    raise ValueError("execution invocation identity must be unique")
                seen.add(event.invocation_id)
                if event.phase == "started":
                    opened[event.invocation_id] = event.tool_name
                elif event.phase == "reused" and not event.origin_invocation_id and self.status == "complete":
                    raise ValueError("complete reused execution evidence requires its origin")
            elif opened.pop(event.invocation_id, None) != event.tool_name:
                raise ValueError("execution completion requires a matching start")
        if self.status == "complete" and opened:
            raise ValueError("complete execution evidence cannot contain unfinished invocations")
        return self
