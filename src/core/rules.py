from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field


class IntentRules(BaseModel):
    docs_patterns: list[str] = Field(default_factory=list)
    save_patterns: list[str] = Field(default_factory=list)
    slack_patterns: list[str] = Field(default_factory=list)


class PlannerRules(BaseModel):
    compare_clause_pattern: str
    docs_identifier_stopwords: list[str] = Field(default_factory=list)


class ValidationRules(BaseModel):
    code_identifier_pattern: str
    keyword_pattern: str
    keyword_stopwords: list[str] = Field(default_factory=list)


class DocsSearchQueryHint(BaseModel):
    identifiers: list[str] = Field(default_factory=list)
    library_name: str
    domains: list[str] = Field(default_factory=list)
    fallback_queries: list[str] = Field(default_factory=list)
    match_mode: Literal["contains", "word"] = "contains"


class DocsSearchRules(BaseModel):
    allowed_doc_path_prefixes: dict[str, list[str]] = Field(default_factory=dict)
    error_page_markers: list[str] = Field(default_factory=list)
    query_hints: list[DocsSearchQueryHint] = Field(default_factory=list)


class RulesConfig(BaseModel):
    intents: IntentRules
    planner: PlannerRules
    validation: ValidationRules
    docs_search: DocsSearchRules


def _default_rules_path() -> Path:
    return Path(__file__).resolve().parents[1] / "infra" / "config" / "agent_rules.toml"


@lru_cache(maxsize=8)
def load_rules_config(path: str | None = None) -> RulesConfig:
    resolved_path = Path(path or os.getenv("RULES_CONFIG_PATH") or _default_rules_path()).resolve()
    payload = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    return RulesConfig.model_validate(payload)


def get_rules_config() -> RulesConfig:
    return load_rules_config(None)
