import unittest

from src.core.answer_schema import AnswerResponse
from src.eval.config_models import BenchmarkCase
from src.eval.metric_rules import compute_rule_scores
from tests.eval.response_fixtures import plain_response, source_hit


class ActionOnlyScoringTest(unittest.TestCase):
    def test_action_only_rule_scores_do_not_require_citations(self) -> None:
        """Delivery-only content needs no citations, including after incidental retrieval."""
        scenarios = [
            ("save-without-search", ["save_text"], []),
            ("slack-without-search", ["slack_notify"], []),
            ("unspecified-action", [], []),
            ("unrelated-search", [], [source_hit()]),
        ]
        for name, tools, hits in scenarios:
            with self.subTest(scenario=name):
                scores = compute_rule_scores(
                    case=BenchmarkCase(
                        case_id=name, category="tool_action", query="이 본문을 전달해줘.",
                        expected_tools=tools,
                    ),
                    response=AnswerResponse.model_validate(plain_response("전달할 본문")),
                    called_tools=tools,
                    observed_hits=hits,
                    runtime_errors=[],
                    response_errors=[],
                    judge_errors=[],
                )

                self.assertEqual(scores["reference_coverage"], 1.0)
                self.assertEqual(scores["citation_traceability"], 1.0)

    def test_live_slack_required_case_needs_delivery_success_for_tool_choice(self) -> None:
        case = BenchmarkCase(
            case_id="tool_action_live_slack",
            category="tool_action",
            query="결과를 slack으로 보내줘",
            expected_tools=["slack_notify"],
        )

        failed_scores = compute_rule_scores(
            case=case,
            response=AnswerResponse.model_validate(plain_response("공유 본문\n\n전송 실패")),
            called_tools=["slack_notify"],
            observed_hits=[],
            runtime_errors=[],
            response_errors=[],
            judge_errors=[],
            slack_delivery_required=True,
            slack_delivery_status="failed",
        )
        success_scores = compute_rule_scores(
            case=case,
            response=AnswerResponse.model_validate(plain_response("공유 본문\n\n전송 완료")),
            called_tools=["slack_notify"],
            observed_hits=[],
            runtime_errors=[],
            response_errors=[],
            judge_errors=[],
            slack_delivery_required=True,
            slack_delivery_status="success",
        )

        self.assertEqual(failed_scores["tool_choice"], 0.0)
        self.assertEqual(success_scores["tool_choice"], 1.0)

    def test_non_live_slack_case_keeps_existing_tool_choice_behavior(self) -> None:
        case = BenchmarkCase(
            case_id="tool_action_non_live_slack",
            category="tool_action",
            query="결과를 slack으로 보내줘",
            expected_tools=["slack_notify"],
        )

        scores = compute_rule_scores(
            case=case,
            response=AnswerResponse.model_validate(plain_response("공유 본문")),
            called_tools=["slack_notify"],
            observed_hits=[],
            runtime_errors=[],
            response_errors=[],
            judge_errors=[],
            slack_delivery_required=False,
            slack_delivery_status="failed",
        )

        self.assertEqual(scores["tool_choice"], 1.0)


if __name__ == "__main__":
    unittest.main()
