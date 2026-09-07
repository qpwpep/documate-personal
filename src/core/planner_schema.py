from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PlannerRouteName = Literal["docs", "upload"]
PLANNER_ROUTES: tuple[PlannerRouteName, ...] = ("docs", "upload")
# Historical benchmark diagnostics still use this value; new plans preserve tasks.
PLANNER_WARNING_DUPLICATE_ROUTE_MERGED = "duplicate_route_merged"


def normalize_planner_output_input(value: Any) -> Any:
    return value.model_dump(mode="python") if isinstance(value, BaseModel) else value


class RetrievalRequirement(BaseModel):
    """What evidence must establish, independently of a provider search string."""

    model_config = ConfigDict(extra="forbid")
    library: str | None = Field(default=None, description="The one library owning this official source; null for upload or an unknown library.")
    symbols: list[str] = Field(default_factory=list, description="Exact API/code symbols to find, preferably fully qualified official names.")
    version: str | None = Field(default=None, description="Explicit requested version; null when unconstrained. Never substitute another version.")
    aspects: list[str] = Field(default_factory=list, description="Literal parameter/code anchors explicitly requested in the user's dialogue. Do not guess allowable values or add requirements from model knowledge. General explanation instructions belong in query.")
    match: Literal["topic", "symbol", "definition"] = Field(default="topic", description="topic for broad explanation, symbol for API/usage references, definition for the implementation of a named uploaded function/class.")

    @field_validator("symbols", "aspects")
    @classmethod
    def clean_terms(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @field_validator("library", "version")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        return (value.strip() or None) if value is not None else None

    @property
    def specified(self) -> bool:
        return bool(self.library or self.symbols or self.version or self.aspects or self.match != "topic")


class PlannedRequirement(RetrievalRequirement):
    """An independently planned subject has one match mode, never mixed symbol roles."""

    symbols: list[str] = Field(default_factory=list, max_length=1,
                              description="At most one primary target. Use separate tasks for independent symbols; calls inspected inside a function belong in its aspects, not additional definitions.")


class RetrievalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: PlannerRouteName
    query: str = Field(..., min_length=1)
    k: int = Field(..., ge=1, le=10)
    requirement: PlannedRequirement = Field(default_factory=PlannedRequirement)
    requirement_id: str = Field(default="", description="A stable short ID for this independent requirement; preserve it during retries.")

    @field_validator("requirement", mode="before")
    @classmethod
    def coerce_requirement(cls, value: Any) -> Any:
        return normalize_planner_output_input(value)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("query must not be empty")
        return trimmed

    @model_validator(mode="after")
    def assign_requirement_identity(self) -> "RetrievalTask":
        if not self.requirement_id.strip():
            identity = {"route": self.route, "requirement": self.requirement.model_dump()}
            if not self.requirement.specified:
                identity["query"] = " ".join(self.query.split())
            self.requirement_id = "req_" + hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        else:
            self.requirement_id = self.requirement_id.strip()
        return self


class PlannerOutput(BaseModel):
    """Independent evidence requirements or an explicit clarification decision."""

    model_config = ConfigDict(extra="forbid")
    use_retrieval: bool = Field(description="Whether a sufficiently resolved request needs retrieved evidence.")
    tasks: list[RetrievalTask] = Field(max_length=8, description="One task per independent source/subject/version requirement. Multiple tasks may share a route. Include upload even if the file is missing; omit excluded sources.")
    clarification_question: str | None = Field(default=None, description="Ask for the missing referent/subject/version when the request cannot be resolved from dialogue. Then use_retrieval=false and tasks=[]. Otherwise null.")

    @model_validator(mode="after")
    def validate_rules(self) -> "PlannerOutput":
        if not self.use_retrieval and self.tasks:
            raise ValueError("tasks must be empty when use_retrieval is false")
        if self.use_retrieval and not self.tasks:
            raise ValueError("tasks must contain at least one requirement when use_retrieval is true")
        if self.clarification_question is not None:
            self.clarification_question = self.clarification_question.strip() or None
        if self.clarification_question and self.use_retrieval:
            raise ValueError("clarification must be resolved before retrieval")
        identities = [task.requirement_id for task in self.tasks]
        if len(set(identities)) != len(identities):
            raise ValueError("each independent requirement must have a unique ID")
        return self

    @classmethod
    def validate_input(cls, value: Any, warnings: list[str] | None = None) -> "PlannerOutput":
        return cls.model_validate(normalize_planner_output_input(value))

    @classmethod
    def fallback(cls) -> "PlannerOutput":
        return cls(use_retrieval=False, tasks=[])
