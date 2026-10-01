import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.eval.config_models import BenchmarkCase, BenchmarkConfig, ScoreWeights, WeightProfiles
from src.eval.io import load_cases_jsonl, load_config


GENERAL = {
    "answer_quality": 0.20,
    "reference_coverage": 0.20,
    "citation_traceability": 0.20,
    "tool_choice": 0.15,
    "format_language": 0.05,
    "llm_judge": 0.20,
}


@pytest.mark.parametrize("key", ["tool_match", "content_constraints", "citation_compliance", "safety_format", "llm_jugde"])
def test_current_full_weights_reject_unknown_and_legacy_axes(key: str) -> None:
    with pytest.raises(ValidationError):
        ScoreWeights.model_validate({**GENERAL, key: 0.2})


@pytest.mark.parametrize("value", [True, False, "0.2", None, [], {}, -0.1, float("nan"), float("inf"), -float("inf")])
def test_full_weights_require_finite_nonnegative_numbers(value: object) -> None:
    with pytest.raises(ValidationError):
        ScoreWeights.model_validate({**GENERAL, "answer_quality": value})


@pytest.mark.parametrize("value", [True, False, "0.2", None, [], {}, -0.1, float("nan"), float("inf"), -float("inf")])
def test_override_requires_finite_nonnegative_numbers(value: object) -> None:
    with pytest.raises(ValidationError):
        BenchmarkCase(case_id="invalid", category="docs_only", query="q", weight_override={"answer_quality": value})


@pytest.mark.parametrize("key", ["tool_match", "content_constraints", "citation_compliance", "safety_format", "llm_jugde"])
def test_override_rejects_unknown_and_legacy_axes(key: str) -> None:
    with pytest.raises(ValidationError):
        BenchmarkCase(case_id="invalid", category="docs_only", query="q", weight_override={key: 0.2})


def test_full_weight_vector_requires_all_six_axes() -> None:
    with pytest.raises(ValidationError):
        ScoreWeights.model_validate({"answer_quality": 1.0})


@pytest.mark.parametrize("values", [{key: 0 for key in GENERAL}, {key: 1e308 for key in GENERAL}])
def test_full_weight_vector_requires_positive_finite_total(values: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        ScoreWeights.model_validate(values)


def test_full_weights_allow_integer_and_relative_values_above_one() -> None:
    weights = ScoreWeights.model_validate({key: 2 for key in GENERAL})
    assert weights.as_dict() == {key: 2.0 for key in GENERAL}


@pytest.mark.parametrize("text", ["[weigths]\nanswer_quality = 1.0\n", "[weights]\nanswer_quality = 1.0\n", "[weights.generl]\nanswer_quality = 1.0\n", "[runtime]\njudge_enabld = false\n"])
def test_loader_rejects_unknown_sections_and_profile_names(tmp_path: Path, text: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("profile", ["general", "action_with_citations", "action_without_citations"])
def test_provided_profile_requires_complete_vector(profile: str) -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate({"weights": {profile: {"answer_quality": 1}}})


def test_only_omitted_profiles_receive_complete_defaults() -> None:
    custom = {**GENERAL, "answer_quality": 2}
    profiles = BenchmarkConfig.model_validate({"weights": {"general": custom}}).weights
    assert profiles.general.answer_quality == 2
    assert profiles.action_with_citations == WeightProfiles().action_with_citations
    assert profiles.action_without_citations == WeightProfiles().action_without_citations
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate({"weights": {"general": {}}})


@pytest.mark.parametrize("axis", ["reference_coverage", "citation_traceability"])
def test_no_citation_profile_rejects_positive_citation_axes(axis: str) -> None:
    profile = WeightProfiles().action_without_citations.as_dict()
    with pytest.raises(ValidationError, match=axis):
        BenchmarkConfig.model_validate({"weights": {"action_without_citations": {**profile, axis: 0.1}}})


@pytest.mark.parametrize("payload", [{"llm_judge": True}, {"llm_judge": "0.2"}, {"llm_judge": None}, {"citation_compliance": 0.2}, {"unknown_axis": 0.2}])
def test_jsonl_loader_reports_file_line_case_and_invalid_override(tmp_path: Path, payload: dict[str, object]) -> None:
    path = tmp_path / "cases.jsonl"
    valid = {"case_id": "valid", "category": "docs_only", "query": "q"}
    invalid = {**valid, "case_id": "bad_case", "weight_override": payload}
    path.write_text(json.dumps(valid) + "\n" + json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_cases_jsonl(path)
    message = str(exc_info.value)
    assert "cases.jsonl:2" in message
    assert "bad_case" in message
    assert "weight_override" in message


def test_jsonl_loader_keeps_authored_extensions_outside_weight_contract(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"case_id": "authored", "category": "docs_only", "query": "q", "author_note": "extension", "weight_override": {"llm_judge": 0}}), encoding="utf-8")
    assert load_cases_jsonl(path)[0].weight_override == {"llm_judge": 0}


@pytest.mark.parametrize("value", ['"0.2"', "true", "nan", "inf", "-0.1"])
def test_toml_loader_rejects_invalid_weight_values(tmp_path: Path, value: str) -> None:
    fields = [f"{key} = {value if key == 'answer_quality' else number}" for key, number in GENERAL.items()]
    path = tmp_path / "config.toml"
    path.write_text("[weights.general]\n" + "\n".join(fields), encoding="utf-8")
    with pytest.raises(ValidationError, match="weights.general.answer_quality"):
        load_config(path)


def test_toml_loader_retains_runtime_pricing_and_gate_configuration(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[runtime]\njudge_model = "test-model"\njudge_enabled = false\nrequest_timeout_seconds = 12\n[hard_gates]\ncitation_compliance = 0.91\n[pricing.models."example"]\nprompt_per_1k_usd = 0.2\ncompletion_per_1k_usd = 0.5\n', encoding="utf-8")
    config = load_config(path)
    assert config.weights == WeightProfiles()
    assert (config.judge_model, config.judge_enabled, config.request_timeout_seconds) == ("test-model", False, 12)
    assert config.hard_gates.citation_compliance == 0.91
    assert config.pricing.models["example"].completion_per_1k_usd == 0.5


def test_repository_config_declares_the_three_complete_profiles() -> None:
    path = Path(__file__).resolve().parents[2] / "data/benchmarks/config.toml"
    assert load_config(path).weights == WeightProfiles()
