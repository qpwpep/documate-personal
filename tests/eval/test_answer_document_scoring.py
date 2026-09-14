import unittest

from src.core.answer_schema import ActionReceipt, AnswerDocument, export_answer_text, finalize_answer, text_document
from src.core.contracts.debug import DebugPayload
from src.eval.config_models import BenchmarkCase
from src.eval.metric_rules import score_reference_coverage
from src.eval.online_runner.response_parser import parse_agent_response
from tests.eval.response_fixtures import answer_provenance


class AnswerDocumentScoringTest(unittest.TestCase):
    def test_source_statement_without_refs_has_zero_reference_coverage(self) -> None:
        """A displayed source statement requires actual resolvable references."""
        response = finalize_answer(text_document("검증하지 않은 설명", basis="source"), [])
        score = score_reference_coverage(
            case=BenchmarkCase(case_id="docs", category="docs_only", query="설명"),
            response=response,
            observed_hits=[],
        )
        self.assertEqual(score, 0.0)

    def test_parser_uses_the_canonical_export_and_preserves_public_actions(self) -> None:
        """Evaluation text and receipts come from the same typed response shown in the UI."""
        content = AnswerDocument.model_validate({"blocks": [{
            "type": "code",
            "language": "python",
            "content": {"text": "def run():\n    return 3\n", "basis": "example", "refs": []},
        }]})
        response = finalize_answer(content, [], actions=[ActionReceipt(kind="save_text", status="success", file_path="output/result.txt")])
        parsed = parse_agent_response(self.final_response_data(response.model_dump(mode="json")))
        self.assertEqual(parsed.response_errors, [])
        self.assertEqual(parsed.response, response)
        self.assertEqual(parsed.response_text, export_answer_text(response))
        self.assertEqual(parsed.actions, response.actions)
        self.assertNotIn("output/result.txt", parsed.response_text)

    def test_parser_rejects_the_retired_answer_schema(self) -> None:
        """A legacy answer payload fails explicitly instead of silently losing its body."""
        parsed = parse_agent_response(self.final_response_data({"answer": "old body", "claims": [], "evidence": []}))
        self.assertIsNone(parsed.response)
        self.assertTrue(any("response invalid" in error for error in parsed.response_errors))

    @staticmethod
    def final_response_data(payload: dict) -> dict:
        debug = DebugPayload().model_dump(mode="json")
        if "content" in payload:
            debug["answer_provenance"] = answer_provenance(payload)
        return {"response": payload, "debug": debug}
