from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PlannerRouteName = Literal["docs", "upload", "local"]
PLANNER_ROUTES: tuple[PlannerRouteName, ...] = ("docs", "upload", "local")


def normalize_planner_output_input(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return value


class RetrievalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: PlannerRouteName
    query: str = Field(..., min_length=1)
    k: int = Field(..., ge=1, le=10)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("query must not be empty")
        return trimmed


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_retrieval: bool
    tasks: list[RetrievalTask]

    @model_validator(mode="after")
    def validate_rules(self) -> "PlannerOutput":
        if not self.use_retrieval and self.tasks:
            raise ValueError("tasks must be empty when use_retrieval is false")
        if self.use_retrieval and not self.tasks:
            raise ValueError("tasks must contain at least one route when use_retrieval is true")

        routes = [task.route for task in self.tasks]
        if len(set(routes)) != len(routes):
            raise ValueError("duplicate routes are not allowed in planner tasks")
        return self

    @classmethod
    def validate_input(cls, value: Any) -> "PlannerOutput":
        return cls.model_validate(normalize_planner_output_input(value))

    @classmethod
    def fallback(cls) -> "PlannerOutput":
        return cls(use_retrieval=False, tasks=[])
