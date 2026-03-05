from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


PlannerRoute = Literal["docs", "upload", "local"]


class RetrievalTask(BaseModel):
    route: PlannerRoute
    query: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("query must not be empty")
        return trimmed


class PlannerOutput(BaseModel):
    use_retrieval: bool = False
    tasks: list[RetrievalTask] = Field(default_factory=list)

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
    def fallback(cls) -> "PlannerOutput":
        return cls(use_retrieval=False, tasks=[])
