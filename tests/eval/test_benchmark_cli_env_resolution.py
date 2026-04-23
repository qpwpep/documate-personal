import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.eval.config_models import BenchmarkCase, BenchmarkConfig, BenchmarkLiveSlackConfig
from src.eval.main import command_run
from src.eval.online_runner import run_online_benchmark
from src.infra.settings import DEFAULT_BENCHMARK_CONFIG_PATH, BenchmarkCLIEnvSettings, load_benchmark_cli_env_settings


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self) -> dict:
        return self._payload


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

    @patch("src.eval.main.run_online_benchmark")
    @patch("src.eval.main.get_settings")
    @patch("src.eval.main.load_benchmark_cli_env_settings")
    def test_command_run_prefers_cli_over_benchmark_env(
        self,
        mock_load_benchmark_cli_env_settings,
        mock_get_settings,
        mock_run_online_benchmark,
    ) -> None:
        mock_load_benchmark_cli_env_settings.return_value = BenchmarkCLIEnvSettings(
            endpoint="http://env-endpoint:9300",
            judge_model="gpt-5-env",
            judge_enabled=False,
            live_slack_enabled=True,
            live_slack_channel_id="CENV",
            live_slack_user_id="UENV",
            live_slack_email="env@example.com",
        )
        mock_get_settings.return_value = SimpleNamespace(
            slack_default_user_id="UDEFAULT",
            slack_default_dm_email="default@example.com",
        )
        mock_run_online_benchmark.return_value = (
            Path("output/benchmarks/run"),
            [],
            SimpleNamespace(track="release", overall_passed=True),
        )

        args = SimpleNamespace(
            mode="online",
            endpoint="http://cli-endpoint:9400",
            config=DEFAULT_BENCHMARK_CONFIG_PATH,
            track="release",
            limit=None,
            fixtures=Path("data/benchmarks/fixtures/cases.generated.jsonl"),
            output_root=Path("output/benchmarks"),
            live_slack=False,
            live_slack_channel_id="CCLI",
            live_slack_user_id="UCLI",
            live_slack_email="cli@example.com",
        )

        command_run(args)

        kwargs = mock_run_online_benchmark.call_args.kwargs
        self.assertEqual(kwargs["endpoint"], "http://cli-endpoint:9400")
        self.assertEqual(kwargs["config"].judge_model, "gpt-5-env")
        self.assertFalse(kwargs["config"].judge_enabled)
        self.assertTrue(kwargs["live_slack"].enabled)
        self.assertEqual(kwargs["live_slack"].channel_id, "CCLI")
        self.assertEqual(kwargs["live_slack"].user_id, "UCLI")
        self.assertEqual(kwargs["live_slack"].email, "cli@example.com")

    @patch("src.eval.online_runner.case_runner.requests.post")
    @patch("src.eval.online_runner.case_runner.load_cases_jsonl")
    def test_dotenv_live_slack_settings_rewrite_payload_and_enable_audit_gate(
        self,
        mock_load_cases_jsonl,
        mock_post,
    ) -> None:
        mock_load_cases_jsonl.return_value = [
            BenchmarkCase(
                case_id="tool_live_env",
                category="tool_action",
                query="share this to slack",
                expected_tools=["slack_notify"],
                slack_channel_id="C123BENCH",
            )
        ]
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "shared", "evidence": []},
                "trace": "trace-id",
                "file_path": "",
                "debug": {
                    "schema_version": 3,
                    "observability_status": "ok",
                    "missing_required_debug_fields": [],
                    "tool_calls": ["slack_notify"],
                    "tool_call_count": 1,
                    "token_usage": {},
                    "model_name": None,
                    "models_used": [],
                    "llm_calls": [],
                    "errors": [],
                    "planner_errors": [],
                    "observed_evidence": [],
                    "retry_context": None,
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,
                    "action_results": {
                        "slack_notify": {
                            "status": "ok",
                            "channel_id": "CENVLIVE",
                            "target_type": "Public Channel",
                        }
                    },
                },
            },
        )

        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "BENCHMARK_SLACK_ENABLED=true",
                        "BENCHMARK_SLACK_CHANNEL_ID=CENVLIVE",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            benchmark_env = load_benchmark_cli_env_settings(
                DEFAULT_BENCHMARK_CONFIG_PATH,
                env_path=env_path,
            )
            live_slack = BenchmarkLiveSlackConfig(
                enabled=benchmark_env.live_slack_enabled,
                channel_id=benchmark_env.live_slack_channel_id,
                user_id=benchmark_env.live_slack_user_id,
                email=benchmark_env.live_slack_email,
            )

            _, _, summary = run_online_benchmark(
                fixtures_path=Path("data/benchmarks/fixtures/cases.generated.jsonl"),
                endpoint="http://127.0.0.1:8000",
                config=BenchmarkConfig(judge_enabled=False),
                config_path=DEFAULT_BENCHMARK_CONFIG_PATH,
                output_root=Path(temp_dir) / "output",
                track="release",
                live_slack=live_slack,
            )

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["slack_channel_id"], "CENVLIVE")
        gate = next(gate for gate in summary.gates if gate.name == "slack_delivery_success_rate")
        self.assertEqual(gate.status, "evaluated")
        self.assertEqual(gate.actual, 1.0)


if __name__ == "__main__":
    unittest.main()
