from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.settings import AppSettings
from src.web.app import create_app
from src.web.cleanup import resolve_download_path, validate_upload_file_path


class WebRuntimeModulesTest(unittest.TestCase):
    def test_create_app_lifespan_initializes_state(self) -> None:
        settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
        with patch("src.web.app.get_settings", return_value=settings):
            with TestClient(create_app()) as client:
                self.assertTrue(hasattr(client.app.state, "session_store"))
                self.assertTrue(hasattr(client.app.state, "runtime_cleaner"))
                response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Hello World"})

    def test_resolve_download_path_rejects_traversal(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with self.assertRaises(Exception):
                resolve_download_path(output_dir, "../escape.txt")

    def test_validate_upload_file_path_enforces_session_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            uploads_root = Path(temp_dir) / "uploads"
            target = uploads_root / "session-a" / "sample.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("print('ok')", encoding="utf-8")

            original_cwd = Path.cwd()
            try:
                import os

                os.chdir(temp_dir)
                validated = validate_upload_file_path(str(target), "session-a")
                self.assertEqual(validated, str(target.resolve()))
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
