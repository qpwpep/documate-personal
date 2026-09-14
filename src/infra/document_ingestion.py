"""Local document conversion contracts, independent of the optional Docling runtime."""

from __future__ import annotations

import hashlib
import io
import zipfile
import time
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.core.documents import ParsedDocument
from src.core.uploads import UploadRecord
from src.core.upload_formats import DOCUMENT_MEDIA_TYPES


ADAPTER_VERSION = "2"


class ConversionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifacts_path: str = ""
    ocr_engine: Literal["easyocr", "rapidocr"] = "rapidocr"
    ocr_languages: tuple[str, ...] = ("ko", "en")
    do_ocr: bool = True
    force_full_page_ocr: bool = False
    table_mode: Literal["accurate", "fast"] = "accurate"
    max_pdf_pages: int = Field(default=30, ge=1)
    timeout_seconds: float = Field(default=90, gt=0)
    max_worker_mib: int = Field(default=4096, ge=128)
    max_output_mib: int = Field(default=16, ge=1)
    max_docx_uncompressed_mib: int = Field(default=50, ge=1)
    max_image_pixels: int = Field(default=40_000_000, ge=1)

    def extraction_options(self) -> dict:
        return {"adapter_version": ADAPTER_VERSION, "ocr_engine": self.ocr_engine,
                "ocr_languages": list(self.ocr_languages), "do_ocr": self.do_ocr,
                "force_full_page_ocr": self.force_full_page_ocr, "table_mode": self.table_mode,
                "device": "cpu", "content_scope": "text_and_tables"}


class IngestionError(RuntimeError):
    """A conversion boundary failure with a stable, safe user-facing description."""

    def __init__(self, code: str, message: str, *, file_name: str = "", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.file_name = file_name
        self.retryable = retryable


class DocumentConverterPort(Protocol):
    def convert_many(self, files: Sequence[UploadRecord], *, policy: ConversionPolicy,
                     workspace: Path, cache_dir: Path | None = None,
                     deadline: float | None = None) -> tuple[ParsedDocument, ...]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DocumentIngestionContext:
    converter: DocumentConverterPort | None
    policy: ConversionPolicy
    workspace: Path
    cache_dir: Path | None = None
    cache_max_bytes: int = 32 * 1024 * 1024
    cache_ttl_seconds: int = 1800
    max_chunks: int = 2000
    deadline: float | None = None

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise IngestionError("DOCUMENT_PROCESSING_TIMEOUT", "문서 첨부 처리 시간이 제한을 초과했습니다.", retryable=True)


def read_verified_upload(file: UploadRecord) -> bytes:
    """Bound source reads and verify bytes again before parsing or cache lookup."""
    try:
        with Path(file.path).open("rb") as stream:
            content = stream.read(file.size_bytes + 1)
    except OSError as exc:
        raise IngestionError("DOCUMENT_SOURCE_CHANGED", "변환할 원본 파일을 읽을 수 없습니다.",
                             file_name=file.name) from exc
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    if len(content) != file.size_bytes or digest != file.content_hash:
        raise IngestionError("DOCUMENT_SOURCE_CHANGED", "검증 후 원본 파일이 변경되었습니다.",
                             file_name=file.name)
    return content


def validate_document_input(file: UploadRecord, raw: bytes, policy: ConversionPolicy) -> None:
    """Validate inexpensive format and expansion limits before conversion or a cache hit."""
    suffix = Path(file.name).suffix.casefold()
    try:
        if suffix == ".pdf":
            if b"%PDF-" not in raw[:1024]:
                raise ValueError("PDF signature is missing")
        elif suffix == ".docx":
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.infolist()
                if len(entries) > 2000 or sum(entry.file_size for entry in entries) > policy.max_docx_uncompressed_mib * 1024 * 1024:
                    raise IngestionError("DOCUMENT_LIMIT_EXCEEDED", "DOCX 압축 해제 크기 한도를 초과했습니다.", file_name=file.name)
                if not {"[Content_Types].xml", "word/document.xml"}.issubset(archive.namelist()):
                    raise ValueError("DOCX package is incomplete")
        elif suffix in DOCUMENT_MEDIA_TYPES:
            from PIL import Image

            with Image.open(io.BytesIO(raw)) as picture:
                expected = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".tif": "TIFF",
                            ".tiff": "TIFF", ".webp": "WEBP", ".bmp": "BMP"}[suffix]
                if picture.format != expected:
                    raise ValueError("Image format does not match its name")
                frames = getattr(picture, "n_frames", 1)
                if frames != 1:
                    raise IngestionError("DOCUMENT_INVALID", "여러 프레임 이미지는 페이지별 파일 또는 PDF로 변환해 주세요.", file_name=file.name)
                if picture.width * picture.height > policy.max_image_pixels:
                    raise IngestionError("DOCUMENT_LIMIT_EXCEEDED", "이미지 픽셀 한도를 초과했습니다.", file_name=file.name)
                picture.verify()
        else:
            raise ValueError("Unsupported document type")
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("DOCUMENT_INVALID", "파일 형식이나 내용을 확인해 주세요.", file_name=file.name) from exc
