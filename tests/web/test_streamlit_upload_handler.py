from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

from src.app.web.streamlit_upload_handler import sync_uploaded_file


class _UploadedFile:
    def __init__(self, name: str, payload: bytes | Exception) -> None:
        self.name = name
        self._payload = payload

    def getbuffer(self) -> bytes:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class StreamlitUploadHandlerTest(unittest.TestCase):
    def test_sync_uploaded_file_replaces_same_name_when_bytes_change(self) -> None:
        with TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir)
            existing = session_path / "sample.py"
            existing.write_text("print('ok')", encoding="utf-8")

            result = sync_uploaded_file(
                uploaded_file=_UploadedFile("sample.py", b"print('changed')"),
                session_path=session_path,
                current_file_name="sample.py",
            )

            self.assertTrue(result.changed)
            self.assertEqual(result.file_name, "sample.py")
            self.assertEqual(existing.read_text(encoding="utf-8"), "print('changed')")

    def test_sync_uploaded_file_skips_identical_bytes_at_same_name(self) -> None:
        """An unchanged upload does not trigger a new document revision."""
        with TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir)
            existing = session_path / "sample.py"
            existing.write_bytes(b"value = 3\n")

            result = sync_uploaded_file(_UploadedFile("sample.py", b"value = 3\n"), session_path, "sample.py")

            self.assertFalse(result.changed)
            self.assertEqual(existing.read_bytes(), b"value = 3\n")

    def test_sync_uploaded_file_recreates_missing_current_upload(self) -> None:
        """A current filename alone cannot suppress restoring a missing upload."""
        with TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir)
            result = sync_uploaded_file(_UploadedFile("sample.py", b"value = 3\n"), session_path, "sample.py")

            self.assertTrue(result.changed)
            self.assertEqual((session_path / "sample.py").read_bytes(), b"value = 3\n")

    def test_sync_uploaded_file_replaces_previous_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir)
            old_file = session_path / "old.py"
            old_file.write_text("old", encoding="utf-8")

            result = sync_uploaded_file(
                uploaded_file=_UploadedFile("new.py", b"new content"),
                session_path=session_path,
                current_file_name="old.py",
            )

            self.assertTrue(result.changed)
            self.assertEqual(result.file_name, "new.py")
            self.assertFalse(old_file.exists())
            self.assertEqual((session_path / "new.py").read_bytes(), b"new content")

    def test_sync_uploaded_file_uses_basename_for_uploaded_filename(self) -> None:
        with TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "session"
            session_path.mkdir()
            outside_path = session_path.parent / "evil.py"

            result = sync_uploaded_file(
                uploaded_file=_UploadedFile("../evil.py", b"safe content"),
                session_path=session_path,
                current_file_name=None,
            )

            self.assertTrue(result.changed)
            self.assertEqual(result.file_name, "evil.py")
            self.assertEqual((session_path / "evil.py").read_bytes(), b"safe content")
            self.assertFalse(outside_path.exists())

    def test_sync_uploaded_file_removes_file_when_uploader_cleared(self) -> None:
        with TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir)
            existing = session_path / "sample.py"
            existing.write_text("print('ok')", encoding="utf-8")

            result = sync_uploaded_file(
                uploaded_file=None,
                session_path=session_path,
                current_file_name="sample.py",
            )

            self.assertTrue(result.changed)
            self.assertTrue(result.removed)
            self.assertIsNone(result.file_name)
            self.assertFalse(existing.exists())

    def test_sync_uploaded_file_returns_error_for_invalid_content(self) -> None:
        with TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir)
            old_file = session_path / "old.py"
            old_file.write_text("old", encoding="utf-8")

            result = sync_uploaded_file(
                uploaded_file=_UploadedFile("broken.py", ValueError("bad upload")),
                session_path=session_path,
                current_file_name="old.py",
            )

            self.assertIsNone(result.file_name)
            self.assertFalse(result.changed)
            self.assertEqual(result.error_message, "파일 업로드 실패 (내용 오류): bad upload")
            self.assertEqual(old_file.read_text(encoding="utf-8"), "old")


if __name__ == "__main__":
    unittest.main()


def test_staging_adds_multiple_files_without_replacing_existing_bytes(tmp_path):
    """Staging additions keeps every existing source intact until server confirmation."""
    from src.app.web.streamlit_upload_handler import stage_uploaded_files
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
    from src.app.web.streamlit_upload_handler import stage_uploaded_files
    existing = SimpleNamespace(file_id="original", name="sample.py", size_bytes=3, content_hash="sha256:" + hashlib.sha256(b"old").hexdigest())
    result = stage_uploaded_files([_UploadedFile("sample.py", b"new")], tmp_path, existing_files=[existing], max_files=10, max_file_mib=10, max_total_mib=50)
    assert result.errors == []
    assert result.files[0].conflicting_file_id == "original"
    assert result.files[0].replace_file_id is None


def test_staging_same_name_same_bytes_skips_but_different_name_keeps_file(tmp_path):
    """A duplicate name and content is a no-op, while a distinct filename remains distinct."""
    import hashlib
    from src.app.web.streamlit_upload_handler import stage_uploaded_files
    existing = SimpleNamespace(file_id="original", name="sample.py", size_bytes=3, content_hash="sha256:" + hashlib.sha256(b"old").hexdigest())
    result = stage_uploaded_files([_UploadedFile("sample.py", b"old"), _UploadedFile("other.py", b"old")], tmp_path, existing_files=[existing], max_files=10, max_file_mib=10, max_total_mib=50)
    assert result.errors == []
    assert result.unchanged_names == ["sample.py"]
    assert [item.name for item in result.files] == ["other.py"]


def test_staging_invalid_batch_does_not_publish_partial_files(tmp_path):
    """A failed file prevents the whole staging batch from being sent for indexing."""
    from src.app.web.streamlit_upload_handler import stage_uploaded_files
    result = stage_uploaded_files([_UploadedFile("ok.py", b"ok"), _UploadedFile("bad.py", ValueError("broken"))], tmp_path, existing_files=[], max_files=10, max_file_mib=10, max_total_mib=50)
    assert result.files == []
    assert "bad.py" in result.errors[0]
    assert not list(tmp_path.rglob("*.py"))


def test_staging_enforces_file_count(tmp_path):
    """The entire proposed attachment set must fit the session file count limit."""
    from src.app.web.streamlit_upload_handler import stage_uploaded_files
    result = stage_uploaded_files([_UploadedFile("a.py", b"x"), _UploadedFile("b.py", b"x")], tmp_path, existing_files=[], max_files=1, max_file_mib=10, max_total_mib=50)
    assert result.files == []
    assert result.errors


def test_staging_enforces_combined_size(tmp_path):
    """Individually valid files are rejected together if their total exceeds the session limit."""
    from src.app.web.streamlit_upload_handler import stage_uploaded_files
    result = stage_uploaded_files([_UploadedFile("a.py", b"x" * (1024 * 1024)), _UploadedFile("b.py", b"x")], tmp_path, existing_files=[], max_files=10, max_file_mib=10, max_total_mib=1)
    assert result.files == []
    assert result.errors


def test_staging_rejects_two_different_versions_of_one_name_in_either_order(tmp_path):
    """Conflicting versions in a single selection are ambiguous regardless of selection order."""
    import hashlib
    from src.app.web.streamlit_upload_handler import stage_uploaded_files
    existing = SimpleNamespace(file_id="original", name="sample.py", size_bytes=3, content_hash="sha256:" + hashlib.sha256(b"old").hexdigest())
    for payloads in [(b"old", b"new"), (b"new", b"old")]:
        result = stage_uploaded_files([_UploadedFile("sample.py", payload) for payload in payloads], tmp_path, existing_files=[existing], max_files=10, max_file_mib=10, max_total_mib=50)
        assert result.files == []
        assert result.errors


def test_staging_stops_reading_after_candidate_bytes_exceed_total_limit(tmp_path):
    """An oversized batch keeps existing files intact and does not load later upload buffers."""
    import hashlib
    from src.app.web.streamlit_upload_handler import stage_uploaded_files

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
