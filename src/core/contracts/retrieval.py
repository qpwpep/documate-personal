"""The single internal result contract shared by retrieval tools and execution."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.core.contracts.debug import RetrievalDiagnostic
from src.core.evidence import SearchHit


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[SearchHit]
    diagnostics: RetrievalDiagnostic
