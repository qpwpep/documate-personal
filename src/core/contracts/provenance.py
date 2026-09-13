"""Server-captured answer inputs, separate from current-turn search observations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.evidence import EvidenceRef


BodyKind = Literal[
    "compose", "acknowledge", "copy_input", "transform_input",
    "copy_answer", "transform_answer", "extract", "unresolved",
]


class AnswerSource(BaseModel):
    """The resolved parent's complete adopted citation set, not its old search candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: Literal["previous", "pending"]
    response_hash: str = Field(min_length=1)
    citation_ids: list[str]

    @field_validator("citation_ids")
    @classmethod
    def unique_references(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value) or len(value) != len(set(value)):
            raise ValueError("source citation IDs must be nonblank and unique")
        return value


class AnswerProvenance(BaseModel):
    """The final construction/validation packet and the server-selected source answer.

    A packet also exists for deterministic copies and fallbacks without a model
    call. It must never be reconstructed by merging a conversation's searches.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    body_kind: BodyKind
    response_hash: str = Field(min_length=1)
    source: AnswerSource | None = None
    evidence_packet: list[EvidenceRef]

    @model_validator(mode="after")
    def consistent_inputs(self) -> AnswerProvenance:
        ids = [item.id for item in self.evidence_packet]
        if len(ids) != len(set(ids)):
            raise ValueError("answer evidence packet must contain unique references")
        if self.source is not None and self.body_kind not in {"copy_answer", "transform_answer"}:
            raise ValueError("only answer copy or transformation can select a source answer")
        return self
