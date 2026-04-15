from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TavilyArgs(BaseModel):
    query: str = Field(description="Search query for official documentation.")
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = Field(
        default="basic",
        description="Search depth for Tavily.",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Optional domain whitelist for this query.",
    )
