"""Attachment identities and the versioned session upload API contract."""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, model_validator


def validate_session_id(value: str) -> str:
    """Case variants share one session on both case-sensitive and Windows filesystems."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("session_id must contain 1-128 letters, digits, underscores or hyphens")
    return value.casefold()


def normalized_upload_name(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


class UploadFileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=0)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_uri: str = Field(min_length=1)


class UploadRecord(UploadFileInfo):
    """Internal storage reference; public manifests never expose the disk path."""

    path: str = Field(min_length=1)

    def public_info(self) -> UploadFileInfo:
        return UploadFileInfo.model_validate(self.model_dump(exclude={"path"}))


class UploadContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=0)


class UploadManifest(UploadContext):
    files: list[UploadFileInfo] = Field(default_factory=list)

    def context(self) -> UploadContext:
        return UploadContext(epoch=self.epoch, revision=self.revision)


class UploadAddition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=4096)
    name: str = Field(min_length=1, max_length=255)
    replace_file_id: str | None = Field(default=None, min_length=1, max_length=128)


class UploadSyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: str = Field(min_length=1, max_length=128)
    expected_revision: int = Field(ge=0)
    operation_id: str = Field(min_length=1, max_length=128)
    add: list[UploadAddition] = Field(default_factory=list, max_length=100)
    remove: list[str] = Field(default_factory=list, max_length=100)
    clear: bool = False

    @model_validator(mode="after")
    def validate_operation(self) -> UploadSyncRequest:
        if self.clear and (self.add or self.remove):
            raise ValueError("clear cannot be combined with add or remove")
        if len(set(self.remove)) != len(self.remove) or any(not value for value in self.remove):
            raise ValueError("remove must contain distinct, nonempty file IDs")
        targets = [item.replace_file_id for item in self.add if item.replace_file_id]
        if len(set(targets)) != len(targets) or set(targets).intersection(self.remove):
            raise ValueError("a file cannot be replaced or removed twice in one operation")
        return self


class UploadSyncResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: UploadManifest
    changed: bool
    unchanged_names: list[str] = Field(default_factory=list)
