from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.infra.settings import AppSettings


class SettingsPriorityTest(unittest.TestCase):
    def test_dotenv_overrides_process_environment(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                'CHAT_MODEL="gpt-5.4-nano"\n'
                'OPENAI_API_KEY="dotenv-key"\n'
                'TAVILY_API_KEY="dotenv-tavily"\n',
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"CHAT_MODEL": "gpt-5-nano"}, clear=False):
                settings = AppSettings(_env_file=env_file)

            self.assertEqual(settings.chat_model, "gpt-5.4-nano")

    def test_init_kwargs_still_override_dotenv(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                'CHAT_MODEL="gpt-5.4-nano"\n'
                'OPENAI_API_KEY="dotenv-key"\n'
                'TAVILY_API_KEY="dotenv-tavily"\n',
                encoding="utf-8",
            )

            settings = AppSettings(
                _env_file=env_file,
                chat_model="gpt-5.4-mini",
            )

            self.assertEqual(settings.chat_model, "gpt-5.4-mini")


if __name__ == "__main__":
    unittest.main()
