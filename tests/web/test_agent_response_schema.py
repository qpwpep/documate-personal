import unittest

from pydantic import ValidationError

from src.web.schemas import AgentResponse


class AgentResponseSchemaTest(unittest.TestCase):
    def test_structured_response_payload_is_valid(self) -> None:
        payload = {
            "response": {
                "answer": "hello",
                "evidence": [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://numpy.org/doc/stable/",
                        "url_or_path": "https://numpy.org/doc/stable/",
                        "title": "NumPy Docs",
                        "snippet": "broadcasting rule",
                        "score": 0.99,
                    }
                ],
            },
            "trace": "trace-id",
            "file_path": None,
            "debug": None,
        }
        result = AgentResponse.model_validate(payload)
        self.assertEqual(result.response.answer, "hello")
        self.assertEqual(len(result.response.evidence), 1)

    def test_plain_string_response_is_rejected(self) -> None:
        legacy_payload = {
            "response": "legacy string response",
            "trace": "trace-id",
            "file_path": None,
            "debug": None,
        }
        with self.assertRaises(ValidationError):
            AgentResponse.model_validate(legacy_payload)

    def test_debug_retry_context_is_optional_and_parseable(self) -> None:
        payload = {
            "response": {"answer": "uncertain", "evidence": []},
            "trace": "trace-id",
            "file_path": None,
            "debug": {
                "tool_calls": ["tavily_search"],
                "tool_call_count": 1,
                "errors": ["validate_evidence: retry_reason=no_evidence"],
                "observed_evidence": [],
                "retry_context": {
                    "attempt": 1,
                    "max_retries": 1,
                    "retry_reason": "no_evidence",
                    "retrieval_feedback": "query too narrow",
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                    "score_avg": None,
                },
            },
        }
        result = AgentResponse.model_validate(payload)
        self.assertIsNotNone(result.debug)
        self.assertIsNotNone(result.debug.retry_context)
        self.assertEqual(result.debug.retry_context.retry_reason, "no_evidence")


if __name__ == "__main__":
    unittest.main()
