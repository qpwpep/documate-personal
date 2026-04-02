import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.contracts.boundary.graph import build_graph_state_input
from src.nodes.planner.prompt_builder import build_planner_messages


class PlannerPromptBuilderTest(unittest.TestCase):
    def test_build_planner_messages_keeps_library_level_docs_guidance(self) -> None:
        state = build_graph_state_input(
            user_input="Explain bs4 from official docs.",
            messages=[HumanMessage(content="Explain bs4 from official docs.")],
        )

        messages = build_planner_messages(state)
        system_messages = [
            str(message.content) for message in messages if isinstance(message, SystemMessage)
        ]

        self.assertTrue(
            any(
                "preserve the library/framework name in task.query" in content
                and "bare library-level requests" in content
                for content in system_messages
            )
        )

    def test_build_planner_messages_keeps_recent_context_for_numeric_followup(self) -> None:
        state = build_graph_state_input(
            user_input="1",
            messages=[
                HumanMessage(content="판다스의 성능 최적화를 알려줘."),
                AIMessage(content="공식 문서 기준으로 다음 항목 중 하나를 골라줘: 1. 벡터화 2. dtype 최적화"),
                HumanMessage(content="1"),
            ],
        )

        messages = build_planner_messages(state)
        conversation_messages = [
            message
            for message in messages
            if not isinstance(message, SystemMessage)
        ]

        self.assertEqual(len(conversation_messages), 3)
        self.assertIsInstance(conversation_messages[0], HumanMessage)
        self.assertEqual(conversation_messages[0].content, "판다스의 성능 최적화를 알려줘.")
        self.assertIsInstance(conversation_messages[1], AIMessage)
        self.assertIsInstance(conversation_messages[2], HumanMessage)
        self.assertEqual(conversation_messages[2].content, "1")


if __name__ == "__main__":
    unittest.main()
