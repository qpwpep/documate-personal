import unittest

from src.eval.scoring_rules import score_groundedness
from src.eval.schemas import BenchmarkCase, EvidenceItem


class ToolActionGroundednessTest(unittest.TestCase):
    def test_tool_action_without_evidence_is_not_penalized(self) -> None:
        score = score_groundedness(
            case=BenchmarkCase(case_id="tool-action-1", category="tool_action", query="save this"),
            response_text="전달할 본문\n\n저장 완료: C:\\output\\response.txt",
            response_evidence=[],
            observed_evidence=[],
        )

        self.assertEqual(score, 1.0)

    def test_tool_action_without_response_evidence_keeps_full_score_even_if_observed_evidence_exists(self) -> None:
        score = score_groundedness(
            case=BenchmarkCase(case_id="tool-action-2", category="tool_action", query="share this"),
            response_text="전달할 본문\n\n전송 완료: Slack (C123BENCH)",
            response_evidence=[],
            observed_evidence=[
                EvidenceItem(
                    kind="official",
                    tool="tavily_search",
                    source_id="url:https://example.com/reference",
                    document_id="url:https://example.com/reference",
                    url_or_path="https://example.com/reference",
                    title="reference",
                    snippet="reference snippet",
                    score=0.5,
                )
            ],
        )

        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
