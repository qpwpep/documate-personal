from __future__ import annotations

from collections.abc import MutableSequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PlannerRouteName = Literal["docs", "upload", "local"]
PLANNER_ROUTES: tuple[PlannerRouteName, ...] = ("docs", "upload", "local")
PLANNER_WARNING_DUPLICATE_ROUTE_MERGED = "duplicate_route_merged"


def normalize_planner_output_input(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return value


def _normalize_task_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return value


def _clean_query_fragment(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _merge_query_text(first: Any, second: Any) -> str:
    parts: list[str] = []
    for raw in (first, second):
        cleaned = _clean_query_fragment(raw)
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    return "; ".join(parts)


def _merge_duplicate_route_tasks(tasks: MutableSequence[Any]) -> tuple[list[Any], bool]:
    merged: list[Any] = []
    route_indexes: dict[str, int] = {}
    duplicate_found = False

    for raw_task in tasks:
        task = _normalize_task_payload(raw_task)
        if not isinstance(task, dict):
            merged.append(task)
            continue

        route = str(task.get("route") or "").strip()
        if not route or route not in route_indexes:
            if route:
                route_indexes[route] = len(merged)
            merged.append(dict(task))
            continue

        duplicate_found = True
        existing_index = route_indexes[route]
        existing = dict(merged[existing_index])
        merged_query = _merge_query_text(existing.get("query"), task.get("query"))
        if merged_query:
            existing["query"] = merged_query
        merged[existing_index] = existing

    return merged, duplicate_found


def normalize_duplicate_route_tasks(value: Any) -> tuple[Any, list[str]]:
    payload = normalize_planner_output_input(value)
    if not isinstance(payload, dict):
        return payload, []

    tasks = payload.get("tasks")
    if not isinstance(tasks, MutableSequence):
        return payload, []

    merged_tasks, duplicate_found = _merge_duplicate_route_tasks(tasks)
    if not duplicate_found:
        return payload, []

    normalized_payload = dict(payload)
    normalized_payload["tasks"] = merged_tasks
    return normalized_payload, [PLANNER_WARNING_DUPLICATE_ROUTE_MERGED]


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
    def validate_input(cls, value: Any, warnings: list[str] | None = None) -> "PlannerOutput":
        normalized, normalization_warnings = normalize_duplicate_route_tasks(value)
        if warnings is not None:
            for warning in normalization_warnings:
                if warning not in warnings:
                    warnings.append(warning)
        return cls.model_validate(normalized)

    @classmethod
    def fallback(cls) -> "PlannerOutput":
        return cls(use_retrieval=False, tasks=[])
