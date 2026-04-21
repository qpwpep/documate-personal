import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from src.runtime.nodes.synthesis import make_synthesize_node

from .helpers import _CaptureStructuredSynthesizeLLM, build_legacy_state


class SynthesisPromptBudgetTest(unittest.TestCase):
    def test_synthesize_node_truncates_prompt_evidence_for_small_token_budget(self) -> None:
        capture_llm = _CaptureStructuredSynthesizeLLM(include_raw=True)
        synthesize_node = make_synthesize_node(
            capture_llm,
            verbose=False,
            synthesis_max_tokens=100,
            prompt_snippet_char_limit=400,
        )
        long_snippet = "broadcast " * 300

        _ = synthesize_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                    "retrieved_evidence": [
                        {
                            "kind": "official",
                            "tool": "tavily_search",
                            "source_id": "url:https://numpy.org/doc/stable/",
                            "document_id": "url:https://numpy.org/doc/stable/",
                            "url_or_path": "https://numpy.org/doc/stable/",
                            "title": "NumPy Docs",
                            "snippet": long_snippet,
                            "score": 0.95,
                        }
                    ],
                    "synthesis_attempt": 0,
                }
            )
        )

        evidence_messages = [
            str(message.content)
            for message in (capture_llm.last_messages or [])
            if isinstance(message, SystemMessage) and "[Retrieved Evidence]" in str(message.content)
        ]
        self.assertEqual(len(evidence_messages), 1)
        self.assertIn("source_id: url:https://numpy.org/doc/stable/", evidence_messages[0])
        self.assertNotIn(long_snippet.strip(), evidence_messages[0])
        self.assertLess(len(evidence_messages[0]), 900)


if __name__ == "__main__":
    unittest.main()
