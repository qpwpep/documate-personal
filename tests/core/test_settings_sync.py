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
        self.assertIn("`gpt-5.4-nano`: none, low, medium, high, xhigh", synced)
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
        self.assertEqual(
            {spec.example_group for spec in APP_ENV_SPECS},
            {"required_secrets", "application_settings", "slack"},
        )
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

    def test_generated_memory_settings_stay_in_application_group(self) -> None:
        env_example = build_env_example_text(DEFAULT_BENCHMARK_CONFIG_PATH)
        application_index = env_example.index("# Application settings")
        memory_index = env_example.index("MEMORY_HIGH_WATER_TURNS=8")
        slack_index = env_example.index("# Slack")

        self.assertLess(application_index, memory_index)
        self.assertLess(memory_index, slack_index)
        self.assertEqual(env_example.count("MEMORY_HARD_MAX_BYTES="), 1)


if __name__ == "__main__":
    unittest.main()
