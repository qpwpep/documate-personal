"""Filesystem-independent attachment identities captured by release approval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath, PureWindowsPath

from src.core.uploads import UploadManifest, normalized_upload_name


def logical_upload_name(reference: str) -> str:
    """Interpret a relative fixture reference without resolving a disk location.

    Keep the original reference as the approval key. Only the transport basename
    interprets both supported path separators, independently of the host OS.
    """
    path = PurePosixPath(reference.replace("\\", "/"))
    windows = PureWindowsPath(reference)
    if (not path.name or reference.endswith(("/", "\\")) or path.is_absolute()
            or windows.root or windows.drive or ".." in path.parts or "\x00" in reference):
        raise ValueError(f"unsafe upload reference: {reference}")
    return path.name


def validate_upload_names(references: Sequence[str]) -> None:
    """A scenario must have one unambiguous server name per declared attachment."""
    names: dict[str, str] = {}
    for reference in references:
        key = normalized_upload_name(logical_upload_name(reference))
        if key in names:
            raise ValueError(f"upload name collision: {names[key]} and {reference}")
        names[key] = reference


@dataclass(frozen=True, slots=True)
class ApprovedUpload:
    """The same declared identity and immutable bytes used during approval."""

    logical_ref: str
    content: bytes

    def __post_init__(self) -> None:
        logical_upload_name(self.logical_ref)
        if not isinstance(self.content, bytes):
            raise TypeError("approved upload content must be immutable bytes")

    @property
    def name(self) -> str:
        return logical_upload_name(self.logical_ref)

    @property
    def size(self) -> int:
        return len(self.content)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    def getbuffer(self) -> bytes:
        return self.content


def select_approved_uploads(references: Sequence[str], uploads: Mapping[str, ApprovedUpload]) -> list[ApprovedUpload]:
    """Select a scenario's immutable inputs without a filesystem fallback."""
    validate_upload_names(references)
    selected = []
    for reference in references:
        if reference not in uploads:
            raise ValueError(f"upload fixture missing from approved snapshot: {reference}")
        upload = uploads[reference]
        if upload.logical_ref != reference:
            raise ValueError(f"upload fixture identity differs from approved snapshot key: {reference}")
        selected.append(upload)
    return selected


def verify_upload_manifest(uploads: Sequence[ApprovedUpload], manifest: UploadManifest | None) -> None:
    """Require the server's acknowledged inputs before sending a question."""
    if manifest is None:
        raise ValueError("approved upload manifest is missing")
    expected = {upload.name: (upload.size, "sha256:" + upload.fingerprint) for upload in uploads}
    actual = {item.name: (item.size_bytes, item.content_hash) for item in manifest.files}
    if len(actual) != len(manifest.files):
        raise ValueError("approved upload manifest contains duplicate names")
    if expected.keys() != actual.keys():
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        raise ValueError(f"approved upload manifest names differ: missing={missing}, unexpected={unexpected}")
    for name, identity in expected.items():
        if actual[name] != identity:
            raise ValueError(f"approved upload manifest size or hash differs: {name}")
