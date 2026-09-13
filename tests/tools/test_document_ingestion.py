from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.core.uploads import UploadRecord
from src.infra.document_ingestion import ConversionPolicy, IngestionError, read_verified_upload, validate_document_input


def record(path: Path, data: bytes) -> UploadRecord:
    path.write_bytes(data)
    return UploadRecord(file_id="document", name=path.name, path=str(path), size_bytes=len(data),
                        content_hash="sha256:" + hashlib.sha256(data).hexdigest(),
                        source_uri="upload:///session/document")


def test_verified_document_reads_exact_original_bytes(tmp_path):
    """Conversion consumes the same bytes whose hash the attachment service approved."""
    source = record(tmp_path / "native.pdf", b"%PDF-1.7\nsource")
    assert read_verified_upload(source) == b"%PDF-1.7\nsource"


def test_changed_document_is_rejected_before_conversion(tmp_path):
    """Replacing an approved source cannot reuse its old snapshot or conversion cache."""
    source = record(tmp_path / "native.pdf", b"%PDF-1.7\nsource")
    Path(source.path).write_bytes(b"%PDF-1.7\nedited")
    with pytest.raises(IngestionError) as failure:
        read_verified_upload(source)
    assert failure.value.code == "DOCUMENT_SOURCE_CHANGED"


def test_policy_extraction_identity_tracks_ocr_and_table_options():
    """Extraction changes invalidate caches while operational deadlines do not."""
    policy = ConversionPolicy()
    assert policy.extraction_options() != policy.model_copy(update={"do_ocr": False}).extraction_options()
    assert policy.extraction_options() != policy.model_copy(update={"table_mode": "fast"}).extraction_options()
    assert policy.extraction_options() == policy.model_copy(update={"timeout_seconds": 30}).extraction_options()


def test_wrong_document_extension_does_not_enable_a_different_parser(tmp_path):
    """Filename dispatch cannot send arbitrary bytes into the selected document backend."""
    source = record(tmp_path / "native.pdf", b"this is not a pdf")
    with pytest.raises(IngestionError) as failure:
        validate_document_input(source, b"this is not a pdf", ConversionPolicy())
    assert failure.value.code == "DOCUMENT_INVALID"


def test_docx_expansion_limit_is_checked_before_conversion_or_cache_lookup(tmp_path):
    """A compressed Office file cannot bypass the decoded-content memory budget."""
    import io
    import zipfile
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "types")
        archive.writestr("word/document.xml", "x" * 1_100_000)
    raw = stream.getvalue()
    source = record(tmp_path / "huge.docx", raw)
    with pytest.raises(IngestionError) as failure:
        validate_document_input(source, raw, ConversionPolicy(max_docx_uncompressed_mib=1))
    assert failure.value.code == "DOCUMENT_LIMIT_EXCEEDED"
