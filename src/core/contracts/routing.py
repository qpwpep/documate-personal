from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RoutingSource = Literal[
    "add_user_message",
    "planner",
    "pre_synthesis_validation",
    "post_synthesis_validation",
]
RoutingTarget = Literal[
    "summarize_old_messages",
    "planner",
    "retrieve_dispatch",
    "pre_synthesis_validation",
    "synthesize",
    "action_postprocess",
]


class RoutingDecision(BaseModel):
    """One committed choice of the next node, shared by execution and diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence: int = Field(ge=1)
    source: RoutingSource
    target: RoutingTarget
    reason: str = Field(min_length=1, pattern=r"\S")


def validate_route_decisions(value: Any) -> list[RoutingDecision]:
    """Validate a complete history or an append delta without dropping records."""
    if not isinstance(value, list):
        raise ValueError("route_decisions must be a list")
    return [RoutingDecision.model_validate(item) for item in value]
