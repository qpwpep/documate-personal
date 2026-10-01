from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.eval.config_models import BenchmarkCase, ScoreWeights, WeightProfiles
from src.eval.io import load_cases_jsonl
from src.eval.weighting import compute_composite_quality_score, compute_rule_weighted_score, resolve_case_weights


def _case(**kwargs: object) -> BenchmarkCase:
    return BenchmarkCase.model_validate({"case_id": "example", "category": "docs_only", "query": "q", **kwargs})


@pytest.mark.parametrize("category", ["docs_only", "rag_only", "hybrid"])
@pytest.mark.parametrize("official,local", [(False, False), (False, True), (True, False), (True, True)])
def test_general_cases_keep_general_profile_regardless_of_citation_flags(category: str, official: bool, local: bool) -> None:
    profiles = WeightProfiles()
    case = _case(category=category, require_official_citation=official, require_local_citation=local)
    effective = resolve_case_weights(case=case, profiles=profiles)
    assert effective.profile_id == "general"
    assert effective.values.as_dict() == pytest.approx(profiles.general.as_dict())


@pytest.mark.parametrize("official,local", [(False, True), (True, False), (True, True)])
def test_action_with_either_citation_requirement_selects_citation_profile(official: bool, local: bool) -> None:
    profiles = WeightProfiles()
    case = _case(category="tool_action", require_official_citation=official, require_local_citation=local)
    effective = resolve_case_weights(case=case, profiles=profiles)
    assert effective.profile_id == "action_with_citations"
    assert effective.values.as_dict() == pytest.approx(profiles.action_with_citations.as_dict())


def test_action_without_citations_excludes_both_citation_axes_and_normalizes() -> None:
    effective = resolve_case_weights(case=_case(category="tool_action"), profiles=WeightProfiles())
    assert effective.profile_id == "action_without_citations"
    assert effective.values.as_dict() == pytest.approx({
        "answer_quality": 0.4 / 0.95,
        "reference_coverage": 0,
        "citation_traceability": 0,
        "tool_choice": 0.3 / 0.95,
        "format_language": 0.1 / 0.95,
        "llm_judge": 0.15 / 0.95,
    })


@pytest.mark.parametrize("axis", ["reference_coverage", "citation_traceability"])
def test_action_override_cannot_reenable_citation_axes(axis: str) -> None:
    case = _case(category="tool_action", weight_override={axis: 1.0})
    with pytest.raises(ValueError, match=f"example.*action_without_citations.{axis}.*zero"):
        resolve_case_weights(case=case, profiles=WeightProfiles())


def test_partial_override_replaces_only_declared_relative_values_then_normalizes() -> None:
    case = _case(weight_override={"citation_traceability": 0.5, "llm_judge": 0.1})
    effective = resolve_case_weights(case=case, profiles=WeightProfiles())
    assert effective.values.as_dict() == pytest.approx({
        "answer_quality": 0.2 / 1.2,
        "reference_coverage": 0.2 / 1.2,
        "citation_traceability": 0.5 / 1.2,
        "tool_choice": 0.15 / 1.2,
        "format_language": 0.05 / 1.2,
        "llm_judge": 0.1 / 1.2,
    })


def test_raw_profile_is_not_normalized_before_override() -> None:
    general = {key: 2 for key in WeightProfiles().general.as_dict()}
    profiles = WeightProfiles(general=ScoreWeights.model_validate(general))
    effective = resolve_case_weights(case=_case(weight_override={"llm_judge": 1}), profiles=profiles)
    assert effective.values.llm_judge == pytest.approx(1 / 11)
    assert effective.values.answer_quality == pytest.approx(2 / 11)


def test_uniform_relative_scaling_preserves_final_vector() -> None:
    profiles = WeightProfiles()
    scaled = WeightProfiles(general=ScoreWeights.model_validate({key: value * 10 for key, value in profiles.general.as_dict().items()}))
    original = resolve_case_weights(case=_case(), profiles=profiles)
    actual = resolve_case_weights(case=_case(), profiles=scaled)
    assert actual.values.as_dict() == pytest.approx(original.values.as_dict())


@pytest.mark.parametrize("payload", [{}, {"weight_override": None}, {"weight_override": {}}])
def test_missing_null_and_empty_override_all_mean_no_change(payload: dict[str, object]) -> None:
    profiles = WeightProfiles()
    effective = resolve_case_weights(case=_case(**payload), profiles=profiles)
    assert effective.values.as_dict() == pytest.approx(profiles.general.as_dict())


def test_explicit_zero_disables_one_axis_and_retains_others() -> None:
    case = _case(weight_override={"llm_judge": 0})
    effective = resolve_case_weights(case=case, profiles=WeightProfiles())
    assert effective.values.llm_judge == 0
    assert effective.values.answer_quality == pytest.approx(0.25)
    assert sum(effective.values.as_dict().values()) == pytest.approx(1)
    assert compute_composite_quality_score(0.8, None, effective) is None


def test_action_explicit_zero_citation_override_is_allowed() -> None:
    case = _case(category="tool_action", weight_override={"reference_coverage": 0, "citation_traceability": 0})
    effective = resolve_case_weights(case=case, profiles=WeightProfiles())
    assert effective.values.reference_coverage == effective.values.citation_traceability == 0


@pytest.mark.parametrize("number", [0, 1e308])
def test_merged_vector_requires_positive_finite_sum(number: float) -> None:
    override = {key: number for key in WeightProfiles().general.as_dict()}
    case = _case(weight_override=override)
    with pytest.raises(ValueError, match="example.*weights.general.*|positive finite") as exc_info:
        resolve_case_weights(case=case, profiles=WeightProfiles())
    assert "positive finite" in str(exc_info.value)


def test_resolving_does_not_mutate_profiles_or_another_cases_weights() -> None:
    profiles = WeightProfiles()
    before = profiles.model_dump()
    overridden = resolve_case_weights(case=_case(weight_override={"tool_choice": 3}), profiles=profiles)
    untouched = resolve_case_weights(case=_case(), profiles=profiles)
    assert profiles.model_dump() == before
    assert overridden.values.tool_choice != untouched.values.tool_choice
    assert untouched.values.as_dict() == pytest.approx(before["general"])
    with pytest.raises(FrozenInstanceError):
        overridden.profile_id = "general"
    with pytest.raises(ValidationError):
        overridden.values.tool_choice = 1


@pytest.mark.parametrize("category,requires_citation,expected", [("docs_only", False, 0.775), ("tool_action", True, 0.8325)])
def test_general_and_citation_action_scores_are_preserved(category: str, requires_citation: bool, expected: float) -> None:
    case = _case(category=category, require_official_citation=requires_citation)
    effective = resolve_case_weights(case=case, profiles=WeightProfiles())
    components = {"answer_quality": 0.8, "reference_coverage": 0.7, "citation_traceability": 0.6, "tool_choice": 0.9, "format_language": 1.0}
    rule = compute_rule_weighted_score(components, effective)
    assert compute_composite_quality_score(rule, 0.85, effective) == pytest.approx(expected)
    assert compute_composite_quality_score(rule, None, effective) is None


def test_no_citation_score_omits_previous_automatic_five_percent_contribution() -> None:
    effective = resolve_case_weights(case=_case(category="tool_action"), profiles=WeightProfiles())
    components = {"answer_quality": 0.8, "reference_coverage": 1, "citation_traceability": 1, "tool_choice": 0.9, "format_language": 1}
    rule = compute_rule_weighted_score(components, effective)
    score = compute_composite_quality_score(rule, 0.85, effective)
    old_score = 0.8675
    assert score == pytest.approx((old_score - 0.05) / 0.95)
    without_inapplicable_scores = {**components, "reference_coverage": 0, "citation_traceability": 0}
    assert compute_rule_weighted_score(without_inapplicable_scores, effective) == rule


def test_migrated_seed_keeps_pre_migration_effective_weights_and_score() -> None:
    path = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/cases.seed.jsonl"
    case = next(case for case in load_cases_jsonl(path) if case.case_id == "docs_seed_003")
    assert case.weight_override == {"citation_traceability": 0.35, "llm_judge": 0.1}
    effective = resolve_case_weights(case=case, profiles=WeightProfiles())
    assert effective.values.as_dict() == {
        "answer_quality": 0.19047619047619047,
        "reference_coverage": 0.19047619047619047,
        "citation_traceability": 0.3333333333333333,
        "tool_choice": 0.14285714285714285,
        "format_language": 0.047619047619047616,
        "llm_judge": 0.09523809523809523,
    }
    components = {"answer_quality": 0.8, "reference_coverage": 0.7, "citation_traceability": 0.6, "tool_choice": 0.9, "format_language": 1.0}
    rule = compute_rule_weighted_score(components, effective)
    assert compute_composite_quality_score(rule, 0.85, effective) == pytest.approx(0.7428571428571429)
