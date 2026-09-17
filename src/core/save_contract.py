"""Immutable identity of a save obligation and the exact bytes it delivers."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SaveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SaveOperation(SaveModel):
    operation_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    contract_revision: int = Field(ge=1)
    target_kind: Literal["compose", "copy_input", "transform_input", "copy_answer", "transform_answer", "extract"]
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    answer_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    export_profile: Literal["answer-text-v1/utf-8-sig"] = "answer-text-v1/utf-8-sig"
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    byte_count: int = Field(ge=1)

    @classmethod
    def for_text(cls, content: str, **identity) -> "SaveOperation":
        payload = content.encode("utf-8-sig")
        return cls(**identity, payload_sha256=hashlib.sha256(payload).hexdigest(), byte_count=len(payload))

    def matches_bytes(self, payload: bytes) -> bool:
        return len(payload) == self.byte_count and hashlib.sha256(payload).hexdigest() == self.payload_sha256

    @property
    def binding_sha256(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), ensure_ascii=False,
                             sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class SaveArtifact(SaveModel):
    artifact_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    byte_count: int = Field(ge=1)
    created_at: float = Field(ge=0)
    expires_at: float = Field(gt=0)

    @model_validator(mode="after")
    def valid_location_and_expiry(self) -> "SaveArtifact":
        if self.filename in {".", ".."} or any(c in self.filename for c in ("/", "\\", ":")):
            raise ValueError("artifact filename must be a basename")
        if not self.filename.endswith(".txt") or self.expires_at <= self.created_at:
            raise ValueError("artifact requires a TXT filename and a finite retention interval")
        return self


class SaveManifest(SaveModel):
    version: Literal[1] = 1
    operation: SaveOperation
    artifact: SaveArtifact

    @model_validator(mode="after")
    def bound_bytes(self) -> "SaveManifest":
        if (self.operation.payload_sha256 != self.artifact.sha256
                or self.operation.byte_count != self.artifact.byte_count):
            raise ValueError("artifact bytes must match the save operation")
        return self
