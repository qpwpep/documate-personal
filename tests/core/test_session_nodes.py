import unittest

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from src.core.conversation_memory import ConversationMemoryPolicy, estimate_text_tokens
from src.core.contracts import GraphState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.runtime.nodes.session import keep_recent_messages, make_summarize_node

from .helpers import _CaptureSummaryLLM, build_legacy_state


class SessionNodeTest(unittest.TestCase):
    def test_compiled_graph_removes_evicted_messages_after_summary(self) -> None:
        summarize_llm = _CaptureSummaryLLM()
        summarize_node = make_summarize_node(summarize_llm, verbose=False, max_turns=1)
        builder = StateGraph(GraphState)
        builder.add_node("summarize_old_messages", summarize_node)
        builder.set_entry_point("summarize_old_messages")
        builder.add_edge("summarize_old_messages", END)
        graph = builder.compile()
        messages = [
            HumanMessage(content="old-user", id="old-user"),
            AIMessage(content="old-answer", id="old-answer"),
            ToolMessage(
                content='{"secret":"OLD_TOOL_PAYLOAD"}',
                name="search",
                tool_call_id="old-tool-call",
                id="old-tool",
            ),
            HumanMessage(content="recent-user", id="recent-user"),
            AIMessage(content="recent-answer", id="recent-answer"),
            HumanMessage(content="current-user", id="current-user"),
        ]

        result = graph.invoke(
            build_graph_state_input(user_input="current-user", messages=messages)
        )

        self.assertEqual(
            [str(message.content) for message in result["messages"]],
            ["recent-user", "recent-answer", "current-user"],
        )
        self.assertEqual(result["runtime"].memory_summary, "summary line")

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
        self.assertIsInstance(updates["messages"][0], RemoveMessage)
        self.assertEqual(updates["messages"][0].id, REMOVE_ALL_MESSAGES)
        self.assertEqual(
            [message.content for message in updates["messages"][1:]],
            ["second request", "second answer", "third request"],
        )

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
        self.assertEqual(updates["runtime"].memory_summary, "existing summary")
        self.assertEqual(updates["messages"][0].id, REMOVE_ALL_MESSAGES)
        self.assertEqual(
            [message.content for message in updates["messages"][1:]],
            ["second request", "third request"],
        )

    def test_summary_replaces_previous_memory_when_recompacting(self) -> None:
        summarize_llm = _CaptureSummaryLLM(content="replacement summary")
        summarize_node = make_summarize_node(
            summarize_llm,
            verbose=False,
            max_turns=1,
        )

        updates = summarize_node(
            build_legacy_state(
                {
                    "memory_summary": "PRIOR_FACT: Python 3.12",
                    "messages": [
                        HumanMessage(content="EVICTED_FACT: deployment uses uv"),
                        AIMessage(content="confirmed"),
                        HumanMessage(content="recent"),
                        AIMessage(content="recent answer"),
                        HumanMessage(content="current"),
                    ],
                }
            )
        )

        prompt = "\n".join(
            str(message.content) for message in summarize_llm.last_messages or []
        )
        self.assertIn("PRIOR_FACT: Python 3.12", prompt)
        self.assertIn("EVICTED_FACT: deployment uses uv", prompt)
        self.assertEqual(updates["runtime"].memory_summary, "replacement summary")
        self.assertNotEqual(
            updates["runtime"].memory_summary,
            "PRIOR_FACT: Python 3.12\nreplacement summary",
        )

    def test_summary_failure_and_blank_output_use_a_bounded_fallback(self) -> None:
        class _FailingSummaryLLM:
            def invoke(self, _messages):
                raise TimeoutError("summary timed out")

        policy = ConversationMemoryPolicy(
            high_water_turns=3,
            low_water_turns=2,
            high_water_tokens=1_000,
            low_water_tokens=500,
            high_water_bytes=4_096,
            low_water_bytes=2_048,
            high_water_messages=10,
            low_water_messages=6,
            summary_max_tokens=32,
            summary_max_bytes=128,
            hard_max_bytes=8_192,
        )
        for label, llm in (
            ("exception", _FailingSummaryLLM()),
            ("blank", _CaptureSummaryLLM(content="   ")),
        ):
            with self.subTest(label=label):
                summarize_node = make_summarize_node(
                    llm,
                    verbose=False,
                    policy=policy,
                )
                updates = summarize_node(
                    build_legacy_state(
                        {
                            "memory_summary": "PRIOR_FACT",
                            "messages": [
                                HumanMessage(content="EVICTED_FACT"),
                                ToolMessage(
                                    content="PRIVATE_TOOL_PAYLOAD",
                                    name="search",
                                    tool_call_id="private-tool",
                                ),
                                AIMessage(content="confirmed"),
                                HumanMessage(content="recent"),
                                AIMessage(content="recent answer"),
                                HumanMessage(content="current"),
                            ],
                        }
                    )
                )

                summary = updates["runtime"].memory_summary
                self.assertTrue(summary)
                self.assertLessEqual(estimate_text_tokens(summary), policy.summary_max_tokens)
                self.assertNotIn("PRIVATE_TOOL_PAYLOAD", summary)
                self.assertEqual(updates["debug"].observability_status, "degraded")
                self.assertTrue(
                    any(
                        event.startswith("memory_summary_fallback:")
                        for event in updates["debug"].validation_events
                    )
                )


if __name__ == "__main__":
    unittest.main()
