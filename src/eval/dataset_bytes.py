"""Validate the physical text format used by byte-exact release approvals.

Validation never normalizes content. A caller must hash and use the same bytes
it supplies here; a prior approval cannot authorize a newly normalized file.
"""
from __future__ import annotations

from pathlib import PurePosixPath


TEXT_FORMAT = "utf8-lf-v1"
_TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".py", ".ipynb"})


def validate_text_bytes(name: str, data: bytes) -> list[str]:
    """Check supported release text, leaving binary evidence uninterpreted."""
    if PurePosixPath(name.replace("\\", "/")).suffix.lower() not in _TEXT_SUFFIXES:
        return []
    errors: list[str] = []
    if data.startswith(b"\xef\xbb\xbf"):
        errors.append(f"{name}: {TEXT_FORMAT} requires UTF-8 without a BOM")
    try:
        data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        errors.append(f"{name}: {TEXT_FORMAT} requires valid UTF-8")
    if b"\r" in data:
        errors.append(f"{name}: {TEXT_FORMAT} requires LF line endings without CR bytes")
    return errors
