from __future__ import annotations

import io
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from src.app.service_manager.process_client import start_background_process
from src.infra.logging_utils import EventFormatter, configure_logging


class LoggingUtilsTest(unittest.TestCase):
    def test_configure_logging_refreshes_existing_documate_handler_formatter(self) -> None:
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler._documate_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(EventFormatter("%(levelname)s %(message)s"))

        try:
            root_logger.handlers = [handler]
            configure_logging(level=logging.DEBUG)

            root_logger.debug("message body")
            self.assertRegex(
                stream.getvalue(),
                r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} DEBUG root event=log message body\n$",
            )
        finally:
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)
            handler.close()

    def test_start_background_process_writes_startup_marker(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_path = root / "runtime" / "fastapi.log"
            fake_process = Mock()
            fake_process.poll.return_value = None

            with patch(
                "src.app.service_manager.process_client.subprocess.Popen",
                return_value=fake_process,
            ) as mock_popen, patch("src.app.service_manager.process_client.time.sleep"):
                process = start_background_process(
                    command=["python", "-m", "uvicorn", "src.app.web.app:app"],
                    cwd=root,
                    log_path=log_path,
                )

            self.assertIs(process, fake_process)
            self.assertTrue(log_path.exists())
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("DOCUMATE SERVICE START", log_text)
            self.assertRegex(log_text, r"DOCUMATE SERVICE START \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
            self.assertIn(f"cwd={root}", log_text)
            self.assertIn("command=python -m uvicorn src.app.web.app:app", log_text)
            mock_popen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
