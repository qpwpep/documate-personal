import json
import unittest

import requests

from src.core.answer_schema import ActionReceipt, AnswerDocument, export_answer_text, finalize_answer, text_document
from src.core.contracts.debug import DebugPayload
from src.eval.config_models import BenchmarkCase
from src.eval.metric_rules import score_citation_traceability, score_reference_coverage
from src.eval.online_runner.response_parser import parse_agent_response


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

    def test_action_only_content_does_not_require_search(self) -> None:
        """A source-free delivery body remains valid without retrieval."""
        response = finalize_answer(text_document("공유할 본문"), [])
        case = BenchmarkCase(case_id="action", category="tool_action", query="공유")
        self.assertEqual(score_reference_coverage(case=case, response=response, observed_hits=[]), 1.0)
        self.assertEqual(
            score_citation_traceability(case=case, response=response, observed_hits=[], called_tools=[]),
            1.0,
        )

    def test_parser_uses_the_canonical_export_and_preserves_public_actions(self) -> None:
        """Evaluation text and receipts come from the same typed response shown in the UI."""
        content = AnswerDocument.model_validate({"blocks": [{
            "type": "code",
            "language": "python",
            "content": {"text": "def run():\n    return 3\n", "basis": "example", "refs": []},
        }]})
        response = finalize_answer(content, [], actions=[ActionReceipt(kind="save_text", status="success", file_path="output/result.txt")])
        parsed = parse_agent_response(self.http_response(response.model_dump(mode="json")))
        self.assertEqual(parsed.response_errors, [])
        self.assertEqual(parsed.response, response)
        self.assertEqual(parsed.response_text, export_answer_text(response))
        self.assertEqual(parsed.actions, response.actions)
        self.assertNotIn("output/result.txt", parsed.response_text)

    def test_parser_rejects_the_retired_answer_schema(self) -> None:
        """A legacy answer payload fails explicitly instead of silently losing its body."""
        parsed = parse_agent_response(self.http_response({"answer": "old body", "claims": [], "evidence": []}))
        self.assertIsNone(parsed.response)
        self.assertTrue(any("response invalid" in error for error in parsed.response_errors))

    @staticmethod
    def http_response(payload: dict) -> requests.Response:
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"response": payload, "debug": DebugPayload().model_dump(mode="json")}).encode("utf-8")
        return response
