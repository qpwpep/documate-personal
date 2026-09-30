from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib

import pytest

from src.app.uploads import (
    StagedUpload, build_upload_sync_request, discard_staged_files,
    review_upload_changes, stage_uploaded_files,
)
from src.core.uploads import UploadFileInfo, UploadManifest


def _manifest(*, file_id="original", content=b"old", revision=1):
    return UploadManifest(epoch="epoch-one", revision=revision, files=[UploadFileInfo(
        file_id=file_id, name="sample.py", size_bytes=len(content),
        content_hash="sha256:" + hashlib.sha256(content).hexdigest(),
        source_uri=f"upload:///session-one/{file_id}",
    )])


def test_request_requires_approval_and_detaches_from_staging_draft(tmp_path):
    manifest = _manifest()
    staged = stage_uploaded_files([_UploadedFile("sample.py", b"new")], tmp_path,
        existing_files=manifest.files, max_files=10, max_file_mib=1, max_total_mib=10)
    with pytest.raises(ValueError, match="교체"):
        build_upload_sync_request(manifest, files=staged.files)
    approved = {"original"}
    request = build_upload_sync_request(manifest, files=staged.files, replace_file_ids=approved)
    payload = request.model_dump(mode="json")
    approved.clear()
    staged.files.clear()
    assert request.model_dump(mode="json") == payload
    assert request.add[0].replace_file_id == "original"
    assert request.expected_revision == 1
    assert request.epoch == "epoch-one"
    assert Path(request.add[0].path).read_bytes() == b"new"












class _UploadedFile:
    def __init__(self, name: str, payload: bytes | Exception) -> None:
        self.name = name
        self._payload = payload

    def getbuffer(self) -> bytes:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload




def test_staging_same_name_changed_bytes_requires_explicit_replacement(tmp_path):
    """A name collision is staged as a conflict rather than silently replacing the file."""
    import hashlib
    existing = SimpleNamespace(file_id="original", name="sample.py", size_bytes=3, content_hash="sha256:" + hashlib.sha256(b"old").hexdigest())
    result = stage_uploaded_files([_UploadedFile("sample.py", b"new")], tmp_path, existing_files=[existing], max_files=10, max_file_mib=10, max_total_mib=50)
    assert result.errors == []
    assert result.files[0].conflicting_file_id == "original"












