import unittest

from src.eval.scoring_rules import (
    compute_final_score,
    compute_rule_weighted_score,
    resolve_effective_weights,
)
from src.eval.schemas import CaseWeightOverride, ScoreWeights


class WeightOverrideTest(unittest.TestCase):
    def test_partial_override_merge_and_normalize(self) -> None:
        base = ScoreWeights()
        override = CaseWeightOverride(citation_compliance=0.5, llm_judge=0.1)
        effective, error = resolve_effective_weights(base_weights=base, case_override=override)

        self.assertIsNone(error)
        self.assertAlmostEqual(sum(effective.as_dict().values()), 1.0, places=8)
        self.assertAlmostEqual(effective.citation_compliance, 0.5 / 1.2, places=8)
        self.assertAlmostEqual(effective.llm_judge, 0.1 / 1.2, places=8)

    def test_final_score_consistency_with_llm_on_off(self) -> None:
        base = ScoreWeights()
        effective, error = resolve_effective_weights(base_weights=base, case_override=None)
        self.assertIsNone(error)

        component_scores = {
            "tool_match": 1.0,
            "content_constraints": 1.0,
            "citation_compliance": 1.0,
            "safety_format": 1.0,
        }
        rule_weighted = compute_rule_weighted_score(component_scores, effective)

        llm_off_score = compute_final_score(rule_weighted_score=rule_weighted, llm_judge_score=None, weights=effective)
        llm_on_score = compute_final_score(rule_weighted_score=rule_weighted, llm_judge_score=1.0, weights=effective)

        self.assertAlmostEqual(llm_off_score, 1.0)
        self.assertAlmostEqual(llm_on_score, 1.0)


if __name__ == "__main__":
    unittest.main()
