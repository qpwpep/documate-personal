import unittest
from pathlib import Path

from src.settings import APP_ENV_SPECS, AppSettings, DEFAULT_BENCHMARK_CONFIG_PATH
from src.settings_sync import build_env_example_text, sync_readme_settings_sections


class SettingsSyncTest(unittest.TestCase):
    def test_env_example_matches_generated_content(self) -> None:
        expected = build_env_example_text(DEFAULT_BENCHMARK_CONFIG_PATH)
        actual = Path(".env.example").read_text(encoding="utf-8")
        self.assertEqual(actual, expected)

    def test_readme_settings_sections_match_generated_content(self) -> None:
        actual = Path("README.md").read_text(encoding="utf-8")
        expected = sync_readme_settings_sections(actual, DEFAULT_BENCHMARK_CONFIG_PATH)
        self.assertEqual(actual, expected)

    def test_app_settings_defaults_match_registry(self) -> None:
        settings = AppSettings(_env_file=None)
        defaults = {spec.field_name: spec.default for spec in APP_ENV_SPECS if spec.field_name is not None}
        for field_name, expected_value in defaults.items():
            self.assertEqual(getattr(settings, field_name), expected_value)


if __name__ == "__main__":
    unittest.main()
