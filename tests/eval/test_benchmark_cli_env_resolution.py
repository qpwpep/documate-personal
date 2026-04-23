import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.infra.settings import DEFAULT_BENCHMARK_CONFIG_PATH, load_benchmark_cli_env_settings


class BenchmarkCLIEnvResolutionTest(unittest.TestCase):
    def test_load_benchmark_cli_env_settings_reads_dotenv_when_os_env_is_empty(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        'BENCHMARK_ENDPOINT="http://env-file:9000"',
                        "JUDGE_MODEL=gpt-5-mini-env",
                        "BENCHMARK_JUDGE_ENABLED=false",
                        "BENCHMARK_SLACK_ENABLED=true",
                        "BENCHMARK_SLACK_CHANNEL_ID=CENVFILE",
                        "BENCHMARK_SLACK_USER_ID=UENVFILE",
                        "BENCHMARK_SLACK_EMAIL=env@example.com",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=True):
                settings = load_benchmark_cli_env_settings(
                    DEFAULT_BENCHMARK_CONFIG_PATH,
                    env_path=env_path,
                )

        self.assertEqual(settings.endpoint, "http://env-file:9000")
        self.assertEqual(settings.judge_model, "gpt-5-mini-env")
        self.assertFalse(settings.judge_enabled)
        self.assertTrue(settings.live_slack_enabled)
        self.assertEqual(settings.live_slack_channel_id, "CENVFILE")
        self.assertEqual(settings.live_slack_user_id, "UENVFILE")
        self.assertEqual(settings.live_slack_email, "env@example.com")

    def test_load_benchmark_cli_env_settings_prefers_dotenv_over_os_env(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        'BENCHMARK_ENDPOINT="http://dotenv:9100"',
                        "JUDGE_MODEL=gpt-5-dotenv",
                        "BENCHMARK_JUDGE_ENABLED=true",
                        "BENCHMARK_SLACK_ENABLED=true",
                        "BENCHMARK_SLACK_CHANNEL_ID=CDOTENV",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "BENCHMARK_ENDPOINT": "http://os-env:9200",
                    "JUDGE_MODEL": "gpt-5-os",
                    "BENCHMARK_JUDGE_ENABLED": "false",
                    "BENCHMARK_SLACK_ENABLED": "false",
                    "BENCHMARK_SLACK_CHANNEL_ID": "COSENV",
                },
                clear=True,
            ):
                settings = load_benchmark_cli_env_settings(
                    DEFAULT_BENCHMARK_CONFIG_PATH,
                    env_path=env_path,
                )

        self.assertEqual(settings.endpoint, "http://dotenv:9100")
        self.assertEqual(settings.judge_model, "gpt-5-dotenv")
        self.assertTrue(settings.judge_enabled)
        self.assertTrue(settings.live_slack_enabled)
        self.assertEqual(settings.live_slack_channel_id, "CDOTENV")


if __name__ == "__main__":
    unittest.main()