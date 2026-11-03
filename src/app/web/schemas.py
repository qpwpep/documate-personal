from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.slack_contract import RecipientInput
from src.core.conversation_memory import DEFAULT_QUERY_MAX_CHARS, validate_query_text
from src.core.uploads import UploadContext, validate_session_id


class ClientUploadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted_suffixes: list[str]
    max_files: int = Field(ge=1)
    max_file_bytes: int = Field(ge=1)
    max_total_bytes: int = Field(ge=1)


class ClientQueryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chars: int = Field(ge=1)
    max_utf8_bytes: int = Field(ge=1)


class ClientConfig(BaseModel):
    """Public input limits; server settings and credentials remain private."""
    model_config = ConfigDict(extra="forbid")

    upload: ClientUploadConfig
    query: ClientQueryConfig


class StagedUpload(BaseModel):
    """A received file that has not yet changed the searchable attachment set."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    upload_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class BrowserUploadAddition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    upload_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    name: str = Field(min_length=1, max_length=255)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    replace_file_id: str | None = Field(default=None, min_length=1, max_length=128)


class BrowserUploadSyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    epoch: str = Field(min_length=1, max_length=128)
    expected_revision: int = Field(ge=0)
    operation_id: str = Field(min_length=1, max_length=128)
    add: list[BrowserUploadAddition] = Field(default_factory=list, max_length=100)
    remove: list[str] = Field(default_factory=list, max_length=100)
    clear: bool = False


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=DEFAULT_QUERY_MAX_CHARS)
    session_id: str
    slack_recipient: RecipientInput | None = None
    uploads: UploadContext
    include_debug: bool = False

    @field_validator("session_id")
    @classmethod
    def _validate_session(cls, value: str) -> str:
        return validate_session_id(value)

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        return validate_query_text(value)
