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
    @staticmethod
    def _runtime_settings_payload(records: list[object]) -> dict[str, str]:
        runtime_records = [
            record
            for record in records
            if getattr(record, "event", None) == "fastapi_runtime_settings"
        ]
        if len(runtime_records) != 1:
            raise AssertionError(f"expected exactly one fastapi_runtime_settings log, got {len(runtime_records)}")

        payload: dict[str, str] = {}
        for part in runtime_records[0].getMessage().split():
            key, value = part.split("=", 1)
            payload[key] = value
        return payload

    def test_create_app_lifespan_initializes_state(self) -> None:
        settings = AppSettings(
            openai_api_key="test-key",
            tavily_api_key="test",
            chat_model="gpt-5.4",
            planner_model="gpt-5.4-mini",
            summary_model="gpt-5.4-nano",
            synthesis_timeout_seconds=42,
            synthesis_max_tokens=2048,
            synthesis_reasoning_effort="high",
        )
        with patch("src.web.app.get_settings", return_value=settings):
            with self.assertLogs("uvicorn", level="INFO") as captured_logs:
                with TestClient(create_app()) as client:
                    self.assertTrue(hasattr(client.app.state, "session_store"))
                    self.assertTrue(hasattr(client.app.state, "runtime_cleaner"))
                    self.assertTrue(hasattr(client.app.state, "agent_request_service"))
                    response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Hello World"})
        self.assertEqual(
            self._runtime_settings_payload(captured_logs.records),
            {
                "chat_model": "gpt-5.4",
                "planner_model": "gpt-5.4-mini",
                "summary_model": "gpt-5.4-nano",
                "synthesis_timeout_seconds": "42",
                "synthesis_max_tokens": "2048",
                "synthesis_reasoning_effort": "high",
            },
        )

    def test_create_app_logs_model_default_reasoning_effort(self) -> None:
        settings = AppSettings(
            openai_api_key="test-key",
            tavily_api_key="test",
            synthesis_reasoning_effort=None,
        )
        with patch("src.web.app.get_settings", return_value=settings):
            with self.assertLogs("uvicorn", level="INFO") as captured_logs:
                with TestClient(create_app()):
                    pass

        payload = self._runtime_settings_payload(captured_logs.records)
        self.assertEqual(payload["synthesis_reasoning_effort"], "model_default")

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
