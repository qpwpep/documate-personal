import unittest

from src.core.answer_schema import finalize_answer, text_document
from src.eval.metric_rules import score_reference_coverage
from src.eval.config_models import BenchmarkCase
from tests.eval.response_fixtures import source_hit


class ToolActionGroundednessTest(unittest.TestCase):
    def test_action_body_does_not_inherit_unrelated_search_requirements(self) -> None:
        """Incidental retrieval does not impose citations on a delivery-only body."""
        response = finalize_answer(text_document("전달할 본문"), [])
        case = BenchmarkCase(case_id="tool-action", category="tool_action", query="share this")
        self.assertEqual(score_reference_coverage(case=case, response=response, observed_hits=[source_hit()]), 1.0)
