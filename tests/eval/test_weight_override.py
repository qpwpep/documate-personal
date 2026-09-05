import unittest

from src.eval.weighting import (
    compute_composite_quality_score,
    compute_rule_weighted_score,
    resolve_base_weights_for_case,
    resolve_effective_weights,
)
from src.eval.config_models import BenchmarkCase, CaseWeightOverride, ScoreWeights


class WeightOverrideTest(unittest.TestCase):
    def test_legacy_weight_keys_map_to_new_score_axes(self) -> None:
        weights = ScoreWeights.model_validate(
            {
                "tool_match": 0.3,
                "content_constraints": 0.25,
                "citation_compliance": 0.2,
                "safety_format": 0.05,
                "llm_judge": 0.2,
            }
        )
        override = CaseWeightOverride.model_validate(
            {
                "tool_match": 0.4,
                "citation_compliance": 0.3,
            }
        )

        self.assertEqual(weights.tool_choice, 0.3)
        self.assertEqual(weights.answer_quality, 0.25)
        self.assertEqual(weights.citation_traceability, 0.2)
        self.assertEqual(weights.format_language, 0.05)
        self.assertEqual(override.as_partial_dict()["tool_choice"], 0.4)
        self.assertEqual(override.as_partial_dict()["citation_traceability"], 0.3)

    def test_partial_override_merge_and_normalize(self) -> None:
        base = ScoreWeights()
        override = CaseWeightOverride(citation_traceability=0.5, llm_judge=0.1)
        effective, error = resolve_effective_weights(base_weights=base, case_override=override)

        self.assertIsNone(error)
        self.assertAlmostEqual(sum(effective.as_dict().values()), 1.0, places=8)
        self.assertAlmostEqual(effective.citation_traceability, 0.5 / 1.2, places=8)
        self.assertAlmostEqual(effective.llm_judge, 0.1 / 1.2, places=8)

    def test_composite_quality_score_consistency_with_llm_on_off(self) -> None:
        base = ScoreWeights()
        effective, error = resolve_effective_weights(base_weights=base, case_override=None)
        self.assertIsNone(error)

        component_scores = {
            "answer_quality": 1.0,
            "groundedness": 1.0,
            "citation_traceability": 1.0,
            "tool_choice": 1.0,
            "format_language": 1.0,
        }
        rule_weighted = compute_rule_weighted_score(component_scores, effective)

        llm_off_score = compute_composite_quality_score(rule_weighted_score=rule_weighted, llm_judge_score=None, weights=effective)
        llm_on_score = compute_composite_quality_score(rule_weighted_score=rule_weighted, llm_judge_score=1.0, weights=effective)

        self.assertAlmostEqual(llm_off_score, 1.0)
        self.assertAlmostEqual(llm_on_score, 1.0)

    def test_tool_action_uses_less_judge_sensitive_base_weights(self) -> None:
        case = BenchmarkCase(
            case_id="tool_action_regression_001",
            category="tool_action",
            query="이번 결과를 팀 채널에 공유해줘.",
            expected_tools=["slack_notify"],
        )

        effective_base = resolve_base_weights_for_case(case=case, base_weights=ScoreWeights())

        self.assertAlmostEqual(effective_base.answer_quality, 0.35)
        self.assertAlmostEqual(effective_base.groundedness, 0.10)
        self.assertAlmostEqual(effective_base.citation_traceability, 0.05)
        self.assertAlmostEqual(effective_base.tool_choice, 0.25)
        self.assertAlmostEqual(effective_base.llm_judge, 0.15)

    def test_tool_action_without_citation_requirements_uses_action_friendly_effective_weights(self) -> None:
        case = BenchmarkCase(
            case_id="tool_action_regression_001",
            category="tool_action",
            query="이번 결과를 팀 채널에 공유해줘.",
            expected_tools=["slack_notify"],
            require_official_citation=False,
            require_local_citation=False,
        )

        effective, error = resolve_effective_weights(
            case=case,
            base_weights=resolve_base_weights_for_case(case=case, base_weights=ScoreWeights()),
            case_override=None,
        )

        self.assertIsNone(error)
        self.assertAlmostEqual(effective.answer_quality, 0.40)
        self.assertAlmostEqual(effective.groundedness, 0.025)
        self.assertAlmostEqual(effective.citation_traceability, 0.025)
        self.assertAlmostEqual(effective.tool_choice, 0.30)
        self.assertAlmostEqual(effective.format_language, 0.10)
        self.assertAlmostEqual(effective.llm_judge, 0.15)


if __name__ == "__main__":
    unittest.main()
