import unittest
from pathlib import Path

from src.infra.settings import APP_ENV_SPECS, APP_ENV_SPEC_BY_NAME, AppSettings, DEFAULT_BENCHMARK_CONFIG_PATH
from src.infra.settings_sync import build_env_example_text, sync_readme_settings_sections


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

    def test_readme_settings_sections_match_generated_content(self) -> None:
        actual = Path("README.md").read_text(encoding="utf-8")
        expected = sync_readme_settings_sections(actual, DEFAULT_BENCHMARK_CONFIG_PATH)
        self.assertEqual(actual, expected)

    def test_readme_settings_sections_document_reasoning_effort_contract(self) -> None:
        actual = Path("README.md").read_text(encoding="utf-8")
        synced = sync_readme_settings_sections(actual, DEFAULT_BENCHMARK_CONFIG_PATH)

        self.assertIn("빈 값이면 모델 기본값", synced)
        self.assertIn("none은 명시 override", synced)
        self.assertIn("`gpt-5.4-nano`: none, low, medium, high, xhigh", synced)
        self.assertIn("`gpt-5-nano`: minimal, low, medium, high", synced)

    def test_app_settings_defaults_match_registry(self) -> None:
        settings = AppSettings(_env_file=None)
        defaults = {spec.field_name: spec.default for spec in APP_ENV_SPECS if spec.field_name is not None}
        for field_name, expected_value in defaults.items():
            self.assertEqual(getattr(settings, field_name), expected_value)


if __name__ == "__main__":
    unittest.main()
