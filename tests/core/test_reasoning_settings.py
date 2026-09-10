from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.infra.settings import APP_ENV_SPECS, AppSettings


@pytest.fixture(autouse=True)
def isolated_settings_environment(monkeypatch):
    for spec in APP_ENV_SPECS:
        monkeypatch.delenv(spec.env_name, raising=False)
    monkeypatch.delenv("PLANNER_REASONING_EFFORT", raising=False)


@pytest.mark.parametrize("field", ["planner_reasoning_effort", "synthesis_reasoning_effort"])
@pytest.mark.parametrize(("value", "expected"), [
    (None, None), ("", None), (" \t ", None), ("default", None),
    (" MODEL_DEFAULT ", None), ("none", "none"), (" NONE ", "none"), (" HiGh ", "high"),
])
def test_reasoning_override_distinguishes_model_default_from_explicit_values(field, value, expected):
    """Both stages normalize default aliases while preserving explicit reasoning choices."""
    settings = AppSettings(_env_file=None, **{field: value})

    assert getattr(settings, field) == expected


@pytest.mark.parametrize("value", ["ultra", "null", False, 3])
def test_planner_rejects_invalid_reasoning_values(value):
    """Invalid planner settings fail validation instead of silently using the model default."""
    with pytest.raises(ValidationError, match="planner_reasoning_effort|PLANNER_REASONING_EFFORT"):
        AppSettings(_env_file=None, planner_reasoning_effort=value)


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6"])
def test_planner_rejects_reasoning_not_supported_by_a_known_model(model):
    """Known GPT-5.6 model names reject minimal even though it is a valid reasoning type value."""
    with pytest.raises(ValidationError, match="PLANNER_REASONING_EFFORT.*minimal.*PLANNER_MODEL") as error:
        AppSettings(_env_file=None, planner_model=model, planner_reasoning_effort="minimal")

    assert model in str(error.value)


def test_planner_model_mismatch_does_not_expose_unrelated_credentials():
    """A rejected model/effort combination reports the offending field without other settings."""
    with pytest.raises(ValidationError) as error:
        AppSettings(
            _env_file=None, openai_api_key="private-key", planner_model="gpt-5.6-luna",
            planner_reasoning_effort="minimal",
        )

    details = error.value.errors(include_context=False, include_url=False)
    assert [(detail["loc"], detail["input"]) for detail in details] == [
        (("planner_reasoning_effort",), "minimal"),
    ]


@pytest.mark.parametrize("model", ["gpt-5-nano", "gpt-5.6-luna-custom"])
def test_unlisted_planner_models_keep_valid_reasoning_for_provider_validation(model):
    """Unlisted names are not assigned another model's capability restrictions."""
    settings = AppSettings(_env_file=None, planner_model=model, planner_reasoning_effort="minimal")

    assert (settings.planner_model, settings.planner_reasoning_effort) == (model, "minimal")


@pytest.mark.parametrize(("dotenv_value", "overrides", "expected"), [
    ("high", {}, "high"),
    ("", {}, None),
    ("high", {"planner_reasoning_effort": "none"}, "none"),
])
def test_planner_reasoning_uses_existing_settings_source_priority(tmp_path, monkeypatch, dotenv_value, overrides, expected):
    """Constructor overrides beat dotenv, and dotenv including a blank value beats process environment."""
    monkeypatch.setenv("PLANNER_REASONING_EFFORT", "low")
    env_file = tmp_path / ".env"
    env_file.write_text(f"PLANNER_REASONING_EFFORT={dotenv_value}\n", encoding="utf-8")

    settings = AppSettings(_env_file=env_file, **overrides)

    assert settings.planner_reasoning_effort == expected
