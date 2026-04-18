import unittest

from src.eval.metric_rules import compute_rule_scores, score_groundedness
from src.eval.schemas import BenchmarkCase


class ActionOnlyScoringTest(unittest.TestCase):
    def test_tool_action_groundedness_is_not_penalized_without_retrieval(self) -> None:
        case = BenchmarkCase(
            case_id="tool_action_regression",
            category="tool_action",
            query="결과를 txt로 저장해줘",
            expected_tools=["save_text"],
        )

        self.assertEqual(
            score_groundedness(
                case=case,
                response_text="저장용 본문\n\n저장 완료: output/response.txt",
                response_evidence=[],
                observed_evidence=[],
            ),
            1.0,
        )

    def test_tool_action_rule_scores_keep_groundedness_at_one(self) -> None:
        case = BenchmarkCase(
            case_id="tool_action_regression",
            category="tool_action",
            query="결과를 slack으로 보내줘",
            expected_tools=["slack_notify"],
        )

        scores = compute_rule_scores(
            case=case,
            response_text="공유용 본문\n\n전송 완료: Slack (C123BENCH)",
            called_tools=["slack_notify"],
            response_evidence=[],
            observed_evidence=[],
            runtime_errors=[],
            response_errors=[],
            judge_errors=[],
        )

        self.assertEqual(scores["groundedness"], 1.0)
        self.assertEqual(scores["citation_traceability"], 1.0)


if __name__ == "__main__":
    unittest.main()
