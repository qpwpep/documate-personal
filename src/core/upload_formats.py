"""Supported attachment formats shared by presentation, validation and ownership."""

from pathlib import Path


CODE_UPLOAD_SUFFIXES = frozenset({".py", ".ipynb"})
DOCUMENT_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".tif": "image/tiff", ".tiff": "image/tiff", ".webp": "image/webp", ".bmp": "image/bmp",
}
ALL_UPLOAD_SUFFIXES = CODE_UPLOAD_SUFFIXES | DOCUMENT_MEDIA_TYPES.keys()


def enabled_upload_suffixes(*, docling_enabled: bool = False) -> frozenset[str]:
    return frozenset(ALL_UPLOAD_SUFFIXES if docling_enabled else CODE_UPLOAD_SUFFIXES)


def is_document_upload(name: str) -> bool:
    return Path(name).suffix.casefold() in DOCUMENT_MEDIA_TYPES
