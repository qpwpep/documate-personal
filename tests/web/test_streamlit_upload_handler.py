from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.web.streamlit_upload_handler import sync_uploaded_file


class _UploadedFile:
    def __init__(self, name: str, payload: bytes | Exception) -> None:
        self.name = name
        self._payload = payload

    def getbuffer(self) -> bytes:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class StreamlitUploadHandlerTest(unittest.TestCase):
    def test_sync_uploaded_file_skips_same_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir)
            existing = session_path / "sample.py"
            existing.write_text("print('ok')", encoding="utf-8")

            result = sync_uploaded_file(
                uploaded_file=_UploadedFile("sample.py", b"print('changed')"),
                session_path=session_path,
                current_file_name="sample.py",
            )

            self.assertFalse(result.changed)
            self.assertEqual(result.file_name, "sample.py")
            self.assertEqual(existing.read_text(encoding="utf-8"), "print('ok')")

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
            self.assertFalse(old_file.exists())


if __name__ == "__main__":
    unittest.main()
