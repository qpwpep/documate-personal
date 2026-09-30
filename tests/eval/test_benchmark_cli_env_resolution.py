from src.core.contracts.usage import LLMCallRecord, TokenUsage
from tests.eval.response_fixtures import answer_provenance, sse_http_response
from src.core.contracts.debug import DEBUG_SCHEMA_VERSION
from tests.eval.response_fixtures import plain_response, slack_action
import unittest

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.eval.config_models import BenchmarkCase, BenchmarkConfig, BenchmarkLiveSlackConfig
from src.eval.io import load_config
from src.eval.main import command_run, resolve_live_slack_dm_recipient
from src.eval.online_runner import run_online_benchmark
from src.eval.pricing import compute_cost_usd
from src.infra.settings import (
    DEFAULT_BENCHMARK_CONFIG_PATH,
    AppSettings,
    BenchmarkCLIEnvSettings,
    load_benchmark_cli_env_settings,
    load_benchmark_env_defaults,
    get_settings,
)


pytestmark = pytest.mark.usefixtures("empty_upload_manifest_http")


@pytest.fixture
def live_cli_case(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "BENCHMARK_SLACK_ENABLED=true\nBENCHMARK_SLACK_CHANNEL_ID=CLIVE\n"
        "SLACK_DEFAULT_USER_ID=UDEFAULT\nSLACK_DEFAULT_DM_EMAIL=default@example.com\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.infra.settings.get_env_file_path", lambda: env_path)
    monkeypatch.setitem(AppSettings.model_config, "env_file", str(env_path))
    for name in ("BENCHMARK_SLACK_USER_ID", "BENCHMARK_SLACK_EMAIL", "SLACK_DEFAULT_USER_ID", "SLACK_DEFAULT_DM_EMAIL"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    config_path = tmp_path / "config.toml"
    config_path.write_text("[runtime]\njudge_enabled = false\n", encoding="utf-8")
    fixtures_path = tmp_path / "cases.jsonl"
    fixtures_path.write_text(BenchmarkCase(
        case_id="channel_only", category="tool_action", query="share this", expected_tools=["slack_notify"],
        slack_recipient={"kind": "channel", "value": "CBENCH"},
    ).model_dump_json() + "\n", encoding="utf-8")
    args = SimpleNamespace(
        mode="online", endpoint="http://benchmark.invalid", config=config_path, track="smoke", limit=None,
        fixtures=fixtures_path, output_root=tmp_path / "output", live_slack=True,
        live_slack_channel_id=None, live_slack_user_id=None, live_slack_email=None,
    )
    payloads = []

    def post(_url, **kwargs):
        payloads.append(kwargs["json"])
        return sse_http_response(200, {"upload_manifest": {"epoch": "fixture-epoch", "revision": 0, "files": []}, "response": {**plain_response("shared"), "actions": [slack_action(channel_id="CLIVE")]}})

    monkeypatch.setattr("src.app.client.requests.post", post)
    yield args, env_path, payloads
    get_settings.cache_clear()


def test_channel_only_cli_run_ignores_invalid_unused_app_dm_default(live_cli_case):
    args, _, payloads = live_cli_case
    assert command_run(args) == 0
    assert len(payloads) == 1
    assert payloads[0]["slack_recipient"] == {"kind": "channel", "value": "CLIVE"}


def test_cli_run_without_slack_ignores_invalid_unused_app_dm_default(live_cli_case):
    args, _, payloads = live_cli_case
    args.fixtures.write_text(BenchmarkCase(
        case_id="no_slack", category="docs_only", query="hello",
    ).model_dump_json() + "\n", encoding="utf-8")
    assert command_run(args) == 0
    assert len(payloads) == 1
    assert "slack_recipient" not in payloads[0]


@pytest.fixture
def live_dm_cli_case(live_cli_case):
    args, env_path, payloads = live_cli_case
    args.fixtures.write_text(BenchmarkCase(
        case_id="dm_only", category="tool_action", query="share this", expected_tools=["slack_notify"],
        slack_recipient={"kind": "user", "value": "UBENCH"},
    ).model_dump_json() + "\n", encoding="utf-8")
    return args, env_path, payloads


def test_cli_dm_case_reports_invalid_app_default_before_http(live_dm_cli_case, empty_upload_manifest_http):
    args, _, payloads = live_dm_cli_case
    with pytest.raises(ValueError, match="SLACK_DEFAULT_USER_ID.*SLACK_DEFAULT_DM_EMAIL.*--live-slack"):
        command_run(args)
    assert payloads == []
    assert empty_upload_manifest_http == []


def test_cli_dm_case_uses_valid_app_default_when_no_override(live_dm_cli_case):
    args, env_path, payloads = live_dm_cli_case
    env_path.write_text(env_path.read_text(encoding="utf-8").replace(
        "SLACK_DEFAULT_DM_EMAIL=default@example.com\n", ""), encoding="utf-8")
    assert command_run(args) == 0
    assert len(payloads) == 1
    assert payloads[0]["slack_recipient"] == {"kind": "user", "value": "UDEFAULT"}


@pytest.mark.parametrize("source", ["cli", "environment"])
@pytest.mark.parametrize("kind, value", [("user", "UEXPLICIT"), ("email", "explicit@example.com")])
def test_explicit_live_dm_bypasses_invalid_app_default(live_dm_cli_case, source, kind, value):
    args, env_path, payloads = live_dm_cli_case
    suffix = "user_id" if kind == "user" else "email"
    if source == "cli":
        setattr(args, "live_slack_" + suffix, value)
    else:
        with env_path.open("a", encoding="utf-8") as env_file:
            env_file.write(f"BENCHMARK_SLACK_{suffix.upper()}={value}\n")
    assert command_run(args) == 0
    assert len(payloads) == 1
    assert payloads[0]["slack_recipient"] == {"kind": kind, "value": value}


@pytest.mark.parametrize("source, user_id, email", [
    ("cli", "UEXPLICIT", "explicit@example.com"),
    ("environment", "UEXPLICIT", "explicit@example.com"),
    ("cli", None, "   "),
    ("environment", None, "not-an-email"),
])
def test_channel_only_cli_run_rejects_invalid_explicit_dm_input(
    live_cli_case, empty_upload_manifest_http, source, user_id, email,
):
    args, env_path, payloads = live_cli_case
    if source == "cli":
        args.live_slack_user_id, args.live_slack_email = user_id, email
    else:
        with env_path.open("a", encoding="utf-8") as env_file:
            for name, value in (("USER_ID", user_id), ("EMAIL", email)):
                if value is not None:
                    env_file.write(f"BENCHMARK_SLACK_{name}={value}\n")
    with pytest.raises(ValueError):
        command_run(args)
    assert payloads == []
    assert empty_upload_manifest_http == []


def test_direct_live_slack_configuration_rejects_channel_as_dm_default():
    with pytest.raises(ValueError, match="DM default must be a user or email"):
        BenchmarkLiveSlackConfig(dm_default={"selector": {"kind": "channel", "value": "C123"}})


@pytest.mark.parametrize("cli, env, expected", [
    ((None, "cli@example.com"), ("UENV", None), ("email", "cli@example.com")),
    ((None, None), (None, "env@example.com"), ("email", "env@example.com")),
    ((None, None), (None, None), ("user", "UDEFAULT")),
])
def test_live_dm_source_precedence_selects_a_whole_recipient(cli, env, expected):
    selector = resolve_live_slack_dm_recipient(
        SimpleNamespace(live_slack_user_id=cli[0], live_slack_email=cli[1]),
        SimpleNamespace(live_slack_user_id=env[0], live_slack_email=env[1]),
    )
    selector = BenchmarkLiveSlackConfig(
        dm_recipient=selector,
        dm_default=AppSettings(_env_file=None, slack_default_user_id="UDEFAULT", slack_default_dm_email=None).slack_default_recipient(),
    ).resolve_dm_recipient()
    assert (selector.kind, selector.value) == expected


@pytest.mark.parametrize("cli, env", [
    (("UCLI", "cli@example.com"), (None, None)),
    ((None, None), ("UENV", "env@example.com")),
    ((None, "   "), ("UENV", None)),
    ((None, None), (None, "not-an-email")),
])
def test_invalid_selected_dm_source_cannot_fall_back(cli, env):
    with pytest.raises(ValueError):
        resolve_live_slack_dm_recipient(
            SimpleNamespace(live_slack_user_id=cli[0], live_slack_email=cli[1]),
            SimpleNamespace(live_slack_user_id=env[0], live_slack_email=env[1]),
        )


class BenchmarkCLIEnvResolutionTest(unittest.TestCase):
    def test_canonical_judge_model_defaults_use_luna(self) -> None:
        """All canonical benchmark configuration entry points select the requested judge model."""
        with TemporaryDirectory() as temp_dir:
            missing_env_path = Path(temp_dir) / ".env"
            with patch.dict("os.environ", {}, clear=True):
                cli_settings = load_benchmark_cli_env_settings(
                    DEFAULT_BENCHMARK_CONFIG_PATH,
                    env_path=missing_env_path,
                )

        self.assertEqual(BenchmarkConfig().judge_model, "gpt-5.6-luna")
        self.assertEqual(load_config(DEFAULT_BENCHMARK_CONFIG_PATH).judge_model, "gpt-5.6-luna")
        self.assertEqual(load_benchmark_env_defaults(DEFAULT_BENCHMARK_CONFIG_PATH)["JUDGE_MODEL"], "gpt-5.6-luna")
        self.assertEqual(cli_settings.judge_model, "gpt-5.6-luna")

    def test_missing_judge_configuration_falls_back_to_luna(self) -> None:
        """Missing config files and omitted runtime settings retain the same judge fallback."""
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            env_path = Path(temp_dir) / ".env"
            with patch.dict("os.environ", {}, clear=True):
                missing_config_settings = load_benchmark_cli_env_settings(config_path, env_path=env_path)
                config_path.write_text("", encoding="utf-8")
                empty_config = load_config(config_path)
                empty_config_settings = load_benchmark_cli_env_settings(config_path, env_path=env_path)

        self.assertEqual(missing_config_settings.judge_model, "gpt-5.6-luna")
        self.assertEqual(empty_config.judge_model, "gpt-5.6-luna")
        self.assertEqual(empty_config_settings.judge_model, "gpt-5.6-luna")

    def test_canonical_pricing_charges_luna_and_preserves_existing_model_rates(self) -> None:
        """Canonical pricing charges mixed current and older calls at their respective model rates."""
        config = load_config(DEFAULT_BENCHMARK_CONFIG_PATH)
        cost = compute_cost_usd(
            llm_calls=[
                LLMCallRecord(stage="synthesis", path="structured", model_name="gpt-5.6-luna",
                              usage=TokenUsage(input_tokens=2000, output_tokens=3000)),
                LLMCallRecord(stage="synthesis", path="structured", model_name="gpt-5.4-nano",
                              usage=TokenUsage(input_tokens=1000, output_tokens=2000)),
            ],
            pricing=config.pricing,
        )

        self.assertAlmostEqual(cost, 0.0067, places=8)

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
            judge_enabled=True,
            live_slack_enabled=True,
            live_slack_channel_id="CENV",
            live_slack_user_id="UENV",
            live_slack_email=None,
        )
        mock_get_settings.return_value = AppSettings(
            _env_file=None,
            slack_default_user_id="UDEFAULT",
            slack_default_dm_email="default@example.com",
        )
        mock_run_online_benchmark.return_value = (
            Path("output/benchmarks/run"),
            [],
            SimpleNamespace(track="release", release_decision=SimpleNamespace(
                scope="release", passed=True, failure_codes=[],
            )),
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
            live_slack_user_id=None,
            live_slack_email="cli@example.com",
        )

        command_run(args)

        kwargs = mock_run_online_benchmark.call_args.kwargs
        self.assertEqual(kwargs["endpoint"], "http://cli-endpoint:9400")
        self.assertEqual(kwargs["config"].judge_model, "gpt-5-env")
        self.assertTrue(kwargs["config"].judge_enabled)
        self.assertTrue(kwargs["live_slack"].enabled)
        self.assertEqual(kwargs["live_slack"].channel_id, "CCLI")
        self.assertEqual(kwargs["live_slack"].dm_recipient.model_dump(), {"kind": "email", "value": "cli@example.com"})

    @patch("src.app.client.requests.post")
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
                slack_recipient={"kind": "channel", "value": "C123BENCH"},
            )
        ]
        mock_post.return_value = sse_http_response(
            200,
            {
                "upload_manifest": {"epoch": "fixture-epoch", "revision": 0, "files": []},
                "response": {**plain_response('shared'), "actions": [slack_action(channel_id="CENVLIVE")]},
                "trace": "trace-id",
                "debug": {
                    "schema_version": DEBUG_SCHEMA_VERSION,
                    "route_decisions": [],
                    "memory_compactions": [],
                    "observability_status": "ok",
                    "missing_required_debug_fields": [],
                    "tool_calls": ["slack_notify"],
                    "tool_call_count": 1,
                    "llm_calls": [],
                    "errors": [],
                    "planner_errors": [],
                    "observed_hits": [],
                    "answer_provenance": answer_provenance(plain_response("shared")),
                    "retry_context": None,
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,

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
            )

            _, _, summary = run_online_benchmark(
                fixtures_path=Path("data/benchmarks/fixtures/cases.generated.jsonl"),
                endpoint="http://127.0.0.1:8000",
                config=BenchmarkConfig(judge_enabled=False),
                config_path=DEFAULT_BENCHMARK_CONFIG_PATH,
                output_root=Path(temp_dir) / "output",
                track="smoke",
                live_slack=live_slack,
            )

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["slack_recipient"], {"kind": "channel", "value": "CENVLIVE"})
        gate = next(gate for gate in summary.gates if gate.name == "slack_delivery_success_rate")
        self.assertEqual(gate.status, "evaluated")
        self.assertEqual(gate.actual, 1.0)


if __name__ == "__main__":
    unittest.main()
