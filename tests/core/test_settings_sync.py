import unittest
from pathlib import Path

from src.infra.settings import APP_ENV_SPECS, APP_ENV_SPEC_BY_NAME, AppSettings, DEFAULT_BENCHMARK_CONFIG_PATH
from src.infra.settings_sync import build_env_example_text, sync_runtime_reference_settings_sections


class SettingsSyncTest(unittest.TestCase):
    def test_env_example_matches_generated_content(self) -> None:
        expected = build_env_example_text(DEFAULT_BENCHMARK_CONFIG_PATH)
        actual = Path(".env.example").read_text(encoding="utf-8")
        self.assertEqual(actual, expected)

    def test_env_example_keeps_reasoning_effort_blank_with_model_notes(self) -> None:
        env_example = build_env_example_text(DEFAULT_BENCHMARK_CONFIG_PATH)
        reasoning_spec = APP_ENV_SPEC_BY_NAME["SYNTHESIS_REASONING_EFFORT"]

        for note in reasoning_spec.sync_notes:
            self.assertIn(f"# {note}\n", env_example)
        self.assertIn("SYNTHESIS_REASONING_EFFORT=\n", env_example)
        self.assertNotIn("SYNTHESIS_REASONING_EFFORT=none", env_example)

    def test_env_example_includes_benchmark_live_slack_settings(self) -> None:
        env_example = build_env_example_text(DEFAULT_BENCHMARK_CONFIG_PATH)

        self.assertIn("BENCHMARK_SLACK_ENABLED=false", env_example)
        self.assertIn("BENCHMARK_SLACK_CHANNEL_ID=", env_example)
        self.assertIn("BENCHMARK_SLACK_USER_ID=", env_example)
        self.assertIn("BENCHMARK_SLACK_EMAIL=", env_example)

    def test_runtime_reference_settings_sections_match_generated_content(self) -> None:
        actual = Path("docs/runtime_reference.md").read_text(encoding="utf-8")
        expected = sync_runtime_reference_settings_sections(actual, DEFAULT_BENCHMARK_CONFIG_PATH)
        self.assertEqual(actual, expected)

    def test_runtime_reference_settings_sections_document_reasoning_effort_contract(self) -> None:
        actual = Path("docs/runtime_reference.md").read_text(encoding="utf-8")
        synced = sync_runtime_reference_settings_sections(actual, DEFAULT_BENCHMARK_CONFIG_PATH)

        self.assertIn("빈 값이면 모델 기본값", synced)
        self.assertIn("none은 명시 override", synced)
        self.assertIn("`gpt-5.6-luna`: none, low, medium, high, xhigh, max", synced)
        self.assertIn("`gpt-5-nano`: minimal, low, medium, high", synced)

    def test_app_settings_defaults_match_registry(self) -> None:
        settings = AppSettings(_env_file=None)
        defaults = {spec.field_name: spec.default for spec in APP_ENV_SPECS if spec.field_name is not None}
        for field_name, expected_value in defaults.items():
            self.assertEqual(getattr(settings, field_name), expected_value)

    def test_env_registry_groups_are_explicit_and_unique(self) -> None:
        env_names = [spec.env_name for spec in APP_ENV_SPECS]
        field_names = [spec.field_name for spec in APP_ENV_SPECS if spec.field_name]

        self.assertEqual(len(env_names), len(set(env_names)))
        self.assertEqual(len(field_names), len(set(field_names)))
        for spec in APP_ENV_SPECS:
            assert spec.field_name is not None
            self.assertEqual(
                AppSettings.model_fields[spec.field_name].alias,
                spec.env_name,
            )

    def test_memory_policy_rejects_low_watermark_at_or_above_high_watermark(self) -> None:
        with self.assertRaises(ValueError):
            AppSettings(
                _env_file=None,
                memory_high_water_tokens=100,
                memory_low_water_tokens=100,
            )

    def test_memory_policy_rejects_limits_that_cannot_hold_one_complete_turn(self) -> None:
        with self.assertRaises(ValueError):
            AppSettings(
                _env_file=None,
                memory_high_water_messages=2,
                memory_low_water_messages=1,
            )

    def test_generated_settings_are_grouped_by_purpose_without_duplicates(self) -> None:
        """The generated template puts every variable in exactly one purpose-specific section."""
        env_example = build_env_example_text(DEFAULT_BENCHMARK_CONFIG_PATH)
        expected = {
            "Required secrets": ["OPENAI_API_KEY", "TAVILY_API_KEY"],
            "Model selection": ["CHAT_MODEL", "PLANNER_MODEL", "SUMMARY_MODEL"],
            "Planning and summary generation": ["SUMMARY_MAX_TOKENS", "PLANNER_MAX_TOKENS"],
            "Answer generation": [
                "SYNTHESIS_TIMEOUT_SECONDS", "SYNTHESIS_USE_RESPONSES_API", "SYNTHESIS_MAX_RETRIES",
                "SYNTHESIS_MAX_TOKENS", "SYNTHESIS_COMPACT_MAX_TOKENS", "SYNTHESIS_PROMPT_SNIPPET_CHARS",
                "SYNTHESIS_COMPACT_PROMPT_SNIPPET_CHARS", "SYNTHESIS_REASONING_EFFORT",
            ],
            "Document search": ["DOCS_SEARCH_TIMEOUT_SECONDS"],
            "Server and logging": ["VERBOSE", "FASTAPI_URL"],
            "Session lifecycle": ["SESSION_TTL_SECONDS", "MAX_ACTIVE_SESSIONS", "SESSION_CLEANUP_INTERVAL_SECONDS"],
            "File retention and cleanup": ["GENERATED_FILE_TTL_SECONDS", "FILE_CLEANUP_INTERVAL_SECONDS"],
            "Conversation memory": [
                "MEMORY_HIGH_WATER_TURNS", "MEMORY_LOW_WATER_TURNS", "MEMORY_HIGH_WATER_TOKENS",
                "MEMORY_LOW_WATER_TOKENS", "MEMORY_HIGH_WATER_BYTES", "MEMORY_LOW_WATER_BYTES",
                "MEMORY_HIGH_WATER_MESSAGES", "MEMORY_LOW_WATER_MESSAGES", "MEMORY_SUMMARY_MAX_TOKENS",
                "MEMORY_SUMMARY_MAX_BYTES", "MEMORY_HARD_MAX_BYTES",
            ],
            "Slack delivery": ["SLACK_BOT_TOKEN", "SLACK_DEFAULT_DM_EMAIL", "SLACK_DEFAULT_USER_ID"],
            "Benchmark / Eval overrides": ["JUDGE_MODEL", "BENCHMARK_ENDPOINT", "BENCHMARK_JUDGE_ENABLED"],
            "Benchmark Slack delivery": [
                "BENCHMARK_SLACK_ENABLED", "BENCHMARK_SLACK_CHANNEL_ID", "BENCHMARK_SLACK_USER_ID", "BENCHMARK_SLACK_EMAIL",
            ],
        }
        actual: dict[str, list[str]] = {}
        section = ""
        for line in env_example.splitlines():
            if line.removeprefix("# ") in expected:
                section = line.removeprefix("# ")
                self.assertNotIn(section, actual)
                actual[section] = []
            elif line and not line.startswith("#"):
                actual.setdefault(section, []).append(line.split("=", 1)[0])
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
