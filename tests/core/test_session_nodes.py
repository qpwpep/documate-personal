import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.nodes.session import keep_recent_messages, make_summarize_node

from .helpers import _CaptureSummaryLLM, build_legacy_state


class SessionNodeTest(unittest.TestCase):
    def test_keep_recent_messages_preserves_human_turn_boundaries(self) -> None:
        messages = [
            HumanMessage(content="u1"),
            AIMessage(content="a1"),
            ToolMessage(content='{"status":"ok"}', name="tavily_search", tool_call_id="tool-1"),
            HumanMessage(content="u2"),
            AIMessage(content="a2"),
            HumanMessage(content="u3"),
        ]

        kept_messages = keep_recent_messages(messages, max_turns=1)

        self.assertEqual(len(kept_messages), 3)
        self.assertIsInstance(kept_messages[0], HumanMessage)
        self.assertEqual(kept_messages[0].content, "u2")
        self.assertEqual([message.content for message in kept_messages], ["u2", "a2", "u3"])

    def test_summarize_node_uses_transcript_without_tool_payloads(self) -> None:
        summarize_llm = _CaptureSummaryLLM()
        summarize_node = make_summarize_node(summarize_llm, verbose=False, max_turns=1)

        updates = summarize_node(
            build_legacy_state(
                {
                    "messages": [
                        HumanMessage(content="first request"),
                        AIMessage(content="saved to output/response-1.txt"),
                        ToolMessage(
                            content='{"results":[{"url":"https://example.com","snippet":"payload"}]}',
                            name="tavily_search",
                            tool_call_id="tool-1",
                        ),
                        HumanMessage(content="second request"),
                        AIMessage(content="second answer"),
                        HumanMessage(content="third request"),
                    ]
                }
            )
        )

        self.assertIsNotNone(summarize_llm.last_messages)
        assert summarize_llm.last_messages is not None
        self.assertEqual(len(summarize_llm.last_messages), 2)
        self.assertIsInstance(summarize_llm.last_messages[0], SystemMessage)
        self.assertIsInstance(summarize_llm.last_messages[1], HumanMessage)
        transcript = str(summarize_llm.last_messages[1].content)
        self.assertIn("user: first request", transcript)
        self.assertIn("assistant: saved to output/response-1.txt", transcript)
        self.assertNotIn("https://example.com", transcript)
        self.assertNotIn("payload", transcript)
        self.assertEqual(updates["runtime"].memory_summary, "summary line")
        self.assertEqual([message.content for message in updates["messages"]], ["second request", "second answer", "third request"])

    def test_summarize_node_skips_llm_when_transcript_is_blank(self) -> None:
        summarize_llm = _CaptureSummaryLLM()
        summarize_node = make_summarize_node(summarize_llm, verbose=False, max_turns=1)

        updates = summarize_node(
            build_legacy_state(
                {
                    "memory_summary": "existing summary",
                    "messages": [
                        HumanMessage(content="   "),
                        ToolMessage(content='{"status":"ok"}', name="save_text", tool_call_id="tool-1"),
                        HumanMessage(content="second request"),
                        HumanMessage(content="third request"),
                    ],
                }
            )
        )

        self.assertIsNone(summarize_llm.last_messages)
        self.assertNotIn("runtime", updates)
        self.assertNotIn("debug", updates)
        self.assertEqual([message.content for message in updates["messages"]], ["second request", "third request"])


if __name__ == "__main__":
    unittest.main()
