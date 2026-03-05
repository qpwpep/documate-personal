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


if __name__ == "__main__":
    unittest.main()
