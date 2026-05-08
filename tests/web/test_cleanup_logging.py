from __future__ import annotations

import logging
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from src.app.web.cleanup import RuntimeCleaner
from src.infra.settings import AppSettings


class _SessionStoreStub:
    def active_session_ids(self) -> set[str]:
        return set()


class RuntimeCleanerLoggingTest(unittest.TestCase):
    @staticmethod
    def _new_cleaner() -> RuntimeCleaner:
        return RuntimeCleaner(
            settings=AppSettings(
                openai_api_key="test-key",
                tavily_api_key="test-key",
                session_ttl_seconds=10,
                generated_file_ttl_seconds=10,
                file_cleanup_interval_seconds=60,
            ),
            session_store=_SessionStoreStub(),  # type: ignore[arg-type]
        )

    @staticmethod
    def _file_cleanup_event_call(mock_log_event: Mock) -> tuple[tuple[object, ...], dict[str, object]]:
        for call_args in mock_log_event.call_args_list:
            args, kwargs = call_args
            if args[2] == "file_cleanup_event":
                return args, kwargs
        raise AssertionError("file_cleanup_event log was not emitted")

    def test_interval_skipped_cleanup_event_logs_at_debug(self) -> None:
        cleaner = self._new_cleaner()
        cleaner._last_file_cleanup_monotonic = 100.0

        with patch("src.app.web.cleanup.time.monotonic", return_value=110.0), patch(
            "src.app.web.cleanup.log_event",
        ) as mock_log_event:
            result = cleaner.run_once(force=False)

        mock_log_event.assert_called_once()
        args, kwargs = self._file_cleanup_event_call(mock_log_event)
        self.assertEqual(args[1], logging.DEBUG)
        self.assertTrue(kwargs["interval_skipped"])
        self.assertTrue(result["interval_skipped"])

    def test_regular_cleanup_without_deletions_or_errors_logs_at_debug(self) -> None:
        cleaner = self._new_cleaner()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with patch("src.app.web.cleanup.time.monotonic", return_value=100.0), patch(
                "src.app.web.cleanup.time.time",
                return_value=100.0,
            ), patch(
                "src.app.web.cleanup.get_uploads_dir",
                return_value=temp_path / "uploads",
            ), patch(
                "src.app.web.cleanup.get_save_text_output_dir",
                return_value=temp_path / "outputs",
            ), patch(
                "src.app.web.cleanup.log_event",
            ) as mock_log_event:
                result = cleaner.run_once(force=False)

        mock_log_event.assert_called_once()
        args, kwargs = self._file_cleanup_event_call(mock_log_event)
        self.assertEqual(args[1], logging.DEBUG)
        self.assertFalse(kwargs["interval_skipped"])
        self.assertEqual(result["upload_dirs_deleted"], 0)
        self.assertEqual(result["generated_files_deleted"], 0)
        self.assertEqual(result["errors"], 0)

    def test_force_cleanup_event_logs_at_info_even_without_changes(self) -> None:
        cleaner = self._new_cleaner()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with patch("src.app.web.cleanup.time.monotonic", return_value=100.0), patch(
                "src.app.web.cleanup.time.time",
                return_value=100.0,
            ), patch(
                "src.app.web.cleanup.get_uploads_dir",
                return_value=temp_path / "uploads",
            ), patch(
                "src.app.web.cleanup.get_save_text_output_dir",
                return_value=temp_path / "outputs",
            ), patch(
                "src.app.web.cleanup.log_event",
            ) as mock_log_event:
                result = cleaner.run_once(force=True)

        mock_log_event.assert_called_once()
        args, kwargs = self._file_cleanup_event_call(mock_log_event)
        self.assertEqual(args[1], logging.INFO)
        self.assertTrue(kwargs["force"])
        self.assertFalse(result["interval_skipped"])

    def test_cleanup_event_with_deleted_upload_dir_logs_at_info(self) -> None:
        cleaner = self._new_cleaner()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            uploads_root = temp_path / "uploads"
            stale_session_dir = uploads_root / "old-session"
            stale_session_dir.mkdir(parents=True)

            with patch("src.app.web.cleanup.time.monotonic", return_value=100.0), patch(
                "src.app.web.cleanup.time.time",
                return_value=100.0,
            ), patch.object(
                RuntimeCleaner,
                "get_latest_mtime_epoch",
                return_value=1.0,
            ), patch(
                "src.app.web.cleanup.get_uploads_dir",
                return_value=uploads_root,
            ), patch(
                "src.app.web.cleanup.get_save_text_output_dir",
                return_value=temp_path / "outputs",
            ), patch(
                "src.app.web.cleanup.log_event",
            ) as mock_log_event:
                result = cleaner.run_once(force=False)

        mock_log_event.assert_called_once()
        args, kwargs = self._file_cleanup_event_call(mock_log_event)
        self.assertEqual(args[1], logging.INFO)
        self.assertEqual(kwargs["upload_dirs_deleted"], 1)
        self.assertEqual(result["upload_dirs_deleted"], 1)

    def test_cleanup_event_with_deleted_generated_file_logs_at_info(self) -> None:
        cleaner = self._new_cleaner()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "outputs"
            output_dir.mkdir()
            stale_file = output_dir / "old.txt"
            stale_file.write_text("old", encoding="utf-8")
            os.utime(stale_file, (0.0, 0.0))

            with patch("src.app.web.cleanup.time.monotonic", return_value=100.0), patch(
                "src.app.web.cleanup.time.time",
                return_value=100.0,
            ), patch(
                "src.app.web.cleanup.get_uploads_dir",
                return_value=temp_path / "uploads",
            ), patch(
                "src.app.web.cleanup.get_save_text_output_dir",
                return_value=output_dir,
            ), patch(
                "src.app.web.cleanup.log_event",
            ) as mock_log_event:
                result = cleaner.run_once(force=False)

        mock_log_event.assert_called_once()
        args, kwargs = self._file_cleanup_event_call(mock_log_event)
        self.assertEqual(args[1], logging.INFO)
        self.assertEqual(kwargs["generated_files_deleted"], 1)
        self.assertEqual(result["generated_files_deleted"], 1)

    def test_cleanup_event_with_errors_logs_at_info(self) -> None:
        cleaner = self._new_cleaner()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            uploads_root = temp_path / "uploads"
            uploads_root.write_text("not a directory", encoding="utf-8")

            with patch("src.app.web.cleanup.time.monotonic", return_value=100.0), patch(
                "src.app.web.cleanup.time.time",
                return_value=100.0,
            ), patch(
                "src.app.web.cleanup.get_uploads_dir",
                return_value=uploads_root,
            ), patch(
                "src.app.web.cleanup.get_save_text_output_dir",
                return_value=temp_path / "outputs",
            ), patch(
                "src.app.web.cleanup.log_event",
            ) as mock_log_event:
                result = cleaner.run_once(force=False)

        args, kwargs = self._file_cleanup_event_call(mock_log_event)
        self.assertEqual(args[1], logging.INFO)
        self.assertEqual(kwargs["errors"], 1)
        self.assertEqual(result["errors"], 1)


if __name__ == "__main__":
    unittest.main()
