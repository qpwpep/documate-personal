import unittest

from src.eval.scoring_rules import (
    compute_final_score,
    compute_rule_weighted_score,
    resolve_effective_weights,
)
from src.eval.schemas import CaseWeightOverride, ScoreWeights


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

    def test_final_score_consistency_with_llm_on_off(self) -> None:
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

        llm_off_score = compute_final_score(rule_weighted_score=rule_weighted, llm_judge_score=None, weights=effective)
        llm_on_score = compute_final_score(rule_weighted_score=rule_weighted, llm_judge_score=1.0, weights=effective)

        self.assertAlmostEqual(llm_off_score, 1.0)
        self.assertAlmostEqual(llm_on_score, 1.0)


if __name__ == "__main__":
    unittest.main()
