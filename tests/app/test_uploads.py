from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from src.app.uploads import build_upload_sync_request, stage_uploaded_files
from src.core.uploads import UploadFileInfo, UploadManifest


def test_request_requires_approval_and_detaches_from_staging_draft(tmp_path):
    manifest = UploadManifest(epoch="epoch-one", revision=1, files=[UploadFileInfo(
        file_id="original", name="sample.py", size_bytes=3,
        content_hash="sha256:" + hashlib.sha256(b"old").hexdigest(),
        source_uri="upload:///session-one/original",
    )])
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
    assert Path(request.add[0].path).read_bytes() == b"new"


class _UploadedFile:
    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._payload = payload

    def getbuffer(self) -> bytes:
        return self._payload
