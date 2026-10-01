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
    assert request.add[0].content_hash == "sha256:" + hashlib.sha256(b"new").hexdigest()
    assert request.expected_revision == 1
    assert request.epoch == "epoch-one"
    assert Path(request.add[0].path).read_bytes() == b"new"


def test_review_keeps_staged_bytes_and_requires_approval_of_current_identity(tmp_path):
    before = _manifest()
    staged = stage_uploaded_files([_UploadedFile("sample.py", b"new")], tmp_path,
        existing_files=before.files, max_files=10, max_file_mib=1, max_total_mib=10)
    original = build_upload_sync_request(before, files=staged.files, replace_file_ids={"original"})
    fresh = _manifest(file_id="replacement", revision=2)
    reviewed, removals = review_upload_changes(fresh, files=staged.files, remove=["gone", "replacement"])
    assert removals == ["replacement"]
    assert reviewed[0].conflicting_file_id == "replacement"
    assert reviewed[0].path == staged.files[0].path
    assert Path(reviewed[0].path).read_bytes() == b"new"
    assert staged.files[0].conflicting_file_id == "original"
    with pytest.raises(ValueError, match="교체"):
        build_upload_sync_request(fresh, files=reviewed, replace_file_ids={"original"})
    reapplied = build_upload_sync_request(fresh, files=reviewed, replace_file_ids={"replacement"})
    assert reapplied.operation_id != original.operation_id
    assert reapplied.expected_revision == 2
    assert reapplied.add[0].replace_file_id == "replacement"
    assert original.expected_revision == 1
    assert original.add[0].replace_file_id == "original"


def test_review_of_already_committed_bytes_builds_noop_but_retains_cleanup_paths(tmp_path):
    staged = stage_uploaded_files([_UploadedFile("sample.py", b"new")], tmp_path,
        existing_files=[], max_files=10, max_file_mib=1, max_total_mib=10)
    fresh = _manifest(content=b"new", revision=2)
    reviewed, removals = review_upload_changes(fresh, files=staged.files, remove=[])
    assert reviewed[0].conflicting_file_id is None
    request = build_upload_sync_request(fresh, files=reviewed, remove=removals)
    assert request.add == []
    assert Path(reviewed[0].path).is_file()
    discard_staged_files(reviewed, tmp_path)
    assert not list(tmp_path.rglob("*.py"))


def test_request_copies_removals_and_uses_a_new_id_for_a_new_intent():
    manifest = _manifest()
    removals = ["original"]
    request = build_upload_sync_request(manifest, remove=removals)
    removals.clear()
    assert request.remove == ["original"]
    cleared = build_upload_sync_request(manifest, clear=True)
    assert cleared.clear
    assert cleared.operation_id != request.operation_id
    with pytest.raises(ValueError):
        build_upload_sync_request(manifest, remove=["original"], clear=True)


@pytest.mark.parametrize("name", ["../sample.py", r"..\sample.py"])
def test_staging_keeps_only_basename_inside_staging(tmp_path, name):
    result = stage_uploaded_files([_UploadedFile(name, b"safe")], tmp_path,
        existing_files=[], max_files=10, max_file_mib=1, max_total_mib=10)
    assert result.errors == []
    assert result.files[0].name == "sample.py"
    assert Path(result.files[0].path).is_relative_to(tmp_path / "staging")
    assert Path(result.files[0].path).read_bytes() == b"safe"


def test_staging_write_failure_discards_the_entire_batch(tmp_path, monkeypatch):
    existing = tmp_path / "existing.py"
    existing.write_bytes(b"confirmed")
    write_bytes = Path.write_bytes
    writes = 0

    def fail_second_write(path, content):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk full")
        return write_bytes(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_second_write)
    result = stage_uploaded_files([_UploadedFile("one.py", b"one"), _UploadedFile("two.py", b"two")],
        tmp_path, existing_files=[], max_files=10, max_file_mib=1, max_total_mib=10)
    assert result.files == []
    assert result.errors == ["파일 저장 실패: disk full"]
    assert not list((tmp_path / "staging").rglob("*"))
    assert existing.read_bytes() == b"confirmed"


class _UploadedFile:
    def __init__(self, name: str, payload: bytes | Exception) -> None:
        self.name = name
        self._payload = payload

    def getbuffer(self) -> bytes:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_staging_adds_multiple_files_without_replacing_existing_bytes(tmp_path):
    """Staging additions keeps every existing source intact until server confirmation."""
    old = tmp_path / "old.py"
    old.write_bytes(b"old")
    result = stage_uploaded_files(
        [_UploadedFile("one.py", b"one"), _UploadedFile("two.py", b"two")], tmp_path,
        existing_files=[], max_files=10, max_file_mib=10, max_total_mib=50,
    )
    assert result.errors == []
    assert [(item.name, Path(item.path).read_bytes()) for item in result.files] == [("one.py", b"one"), ("two.py", b"two")]
    assert all(Path(item.path).is_relative_to(tmp_path / "staging") for item in result.files)
    assert old.read_bytes() == b"old"


def test_staging_same_name_changed_bytes_requires_explicit_replacement(tmp_path):
    """A name collision is staged as a conflict rather than silently replacing the file."""
    import hashlib
    existing = SimpleNamespace(file_id="original", name="sample.py", size_bytes=3, content_hash="sha256:" + hashlib.sha256(b"old").hexdigest())
    result = stage_uploaded_files([_UploadedFile("sample.py", b"new")], tmp_path, existing_files=[existing], max_files=10, max_file_mib=10, max_total_mib=50)
    assert result.errors == []
    assert result.files[0].conflicting_file_id == "original"


def test_staging_same_name_same_bytes_skips_but_different_name_keeps_file(tmp_path):
    """A duplicate name and content is a no-op, while a distinct filename remains distinct."""
    import hashlib
    existing = SimpleNamespace(file_id="original", name="sample.py", size_bytes=3, content_hash="sha256:" + hashlib.sha256(b"old").hexdigest())
    result = stage_uploaded_files([_UploadedFile("sample.py", b"old"), _UploadedFile("other.py", b"old")], tmp_path, existing_files=[existing], max_files=10, max_file_mib=10, max_total_mib=50)
    assert result.errors == []
    assert result.unchanged_names == ["sample.py"]
    assert [item.name for item in result.files] == ["other.py"]


def test_staging_invalid_batch_does_not_publish_partial_files(tmp_path):
    """A failed file prevents the whole staging batch from being sent for indexing."""
    result = stage_uploaded_files([_UploadedFile("ok.py", b"ok"), _UploadedFile("bad.py", ValueError("broken"))], tmp_path, existing_files=[], max_files=10, max_file_mib=10, max_total_mib=50)
    assert result.files == []
    assert "bad.py" in result.errors[0]
    assert not list(tmp_path.rglob("*.py"))


def test_staging_enforces_file_count(tmp_path):
    """The entire proposed attachment set must fit the session file count limit."""
    result = stage_uploaded_files([_UploadedFile("a.py", b"x"), _UploadedFile("b.py", b"x")], tmp_path, existing_files=[], max_files=1, max_file_mib=10, max_total_mib=50)
    assert result.files == []
    assert result.errors


def test_staging_enforces_combined_size(tmp_path):
    """Individually valid files are rejected together if their total exceeds the session limit."""
    result = stage_uploaded_files([_UploadedFile("a.py", b"x" * (1024 * 1024)), _UploadedFile("b.py", b"x")], tmp_path, existing_files=[], max_files=10, max_file_mib=10, max_total_mib=1)
    assert result.files == []
    assert result.errors


def test_staging_rejects_two_different_versions_of_one_name_in_either_order(tmp_path):
    """Conflicting versions in a single selection are ambiguous regardless of selection order."""
    import hashlib
    existing = SimpleNamespace(file_id="original", name="sample.py", size_bytes=3, content_hash="sha256:" + hashlib.sha256(b"old").hexdigest())
    for payloads in [(b"old", b"new"), (b"new", b"old")]:
        result = stage_uploaded_files([_UploadedFile("sample.py", payload) for payload in payloads], tmp_path, existing_files=[existing], max_files=10, max_file_mib=10, max_total_mib=50)
        assert result.files == []
        assert result.errors


def test_staging_stops_reading_after_candidate_bytes_exceed_total_limit(tmp_path):
    """An oversized batch keeps existing files intact and does not load later upload buffers."""
    import hashlib

    class LaterUpload(_UploadedFile):
        was_read = False

        def getbuffer(self):
            self.was_read = True
            return super().getbuffer()

    old = tmp_path / "existing.py"
    old.write_bytes(b"old")
    existing = SimpleNamespace(file_id="original", name=old.name, size_bytes=3,
                               content_hash="sha256:" + hashlib.sha256(b"old").hexdigest())
    later = LaterUpload("later.py", b"later")
    result = stage_uploaded_files([
        _UploadedFile("one.py", b"a" * (512 * 1024)),
        _UploadedFile("two.py", b"b" * (512 * 1024 + 1)),
        later,
    ], tmp_path, existing_files=[existing], max_files=10, max_file_mib=1, max_total_mib=1)

    assert result.files == []
    assert result.errors == ["전체 첨부 크기는 1 MiB 이하여야 합니다."]
    assert old.read_bytes() == b"old"
    assert not (tmp_path / "staging").exists()
    assert later.was_read is False
