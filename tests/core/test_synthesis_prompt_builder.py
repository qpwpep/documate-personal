import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.contracts.boundary.graph import build_graph_state_input
from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages


class SynthesisPromptBuilderTest(unittest.TestCase):
    def test_build_synthesis_messages_compacts_instruction_messages(self) -> None:
        state = build_graph_state_input(
            user_input="Compare official docs with the uploaded notebook and save it.",
            messages=[HumanMessage(content="Compare official docs with the uploaded notebook and save it.")],
            memory_summary="older context",
        )

        messages, history_before, history_after = build_synthesis_messages(
            state=state,
            action_rules=["Produce the final answer content to save now."],
            deduped_evidence=[
                {
                    "kind": "official",
                    "source_id": "url:https://docs.example.com",
                    "url_or_path": "https://docs.example.com",
                    "title": "Official Docs",
                    "snippet": "Official guidance",
                    "score": 0.9,
                },
                {
                    "kind": "local",
                    "source_id": "path:uploads/demo.ipynb#chunk=0",
                    "url_or_path": "uploads/demo.ipynb",
                    "snippet": "Notebook example",
                    "score": 0.8,
                    "code_metadata": {
                        "cell_id": 2,
                        "calls": [
                            {
                                "call_name": "LogisticRegression",
                                "kwargs": {"max_iter": "200"},
                            }
                        ],
                        "option_literals": ["max_iter=200"],
                    },
                },
            ],
            attempt=2,
            max_turns=6,
        )

        self.assertEqual(history_before, 1)
        self.assertEqual(history_after, 1)
        system_messages = [
            str(message.content) for message in messages if isinstance(message, SystemMessage)
        ]
        self.assertIn("[Synthesis Output Template]", system_messages[1])
        self.assertIn("use [] unless Turn Contract required_sections lists section kinds", system_messages[1])
        self.assertIn("Never write placeholder references", system_messages[1])
        self.assertIn("[Selection And Assembly Mode]", system_messages[2])
        turn_messages = [content for content in system_messages if "[Turn Contract]" in content]
        self.assertEqual(len(turn_messages), 1)
        self.assertLess(
            system_messages.index(turn_messages[0]),
            next(
                index
                for index, message in enumerate(system_messages)
                if "[Retrieved Evidence]" in message
            ),
        )
        self.assertTrue(all("[Action Request]" not in content for content in system_messages))
        self.assertEqual(sum(1 for content in system_messages if "[Retrieved Evidence]" in content), 1)
        self.assertTrue(all("[Official Docs Evidence]" not in content for content in system_messages))
        self.assertTrue(all("[Uploaded Code Evidence]" not in content for content in system_messages))
        self.assertIn("action_rules:", turn_messages[0])
        self.assertIn("hybrid_layout=official_docs -> upload/local detail -> comparison", turn_messages[0])
        self.assertIn("upload_code uses local/upload option_literals", turn_messages[0])
        self.assertIn("retry_note=evidence validation failed previously", turn_messages[0])
        evidence_message = next(content for content in system_messages if "[Retrieved Evidence]" in content)
        self.assertIn("candidate_facts: max_iter=200", evidence_message)
        self.assertLess(evidence_message.index("code_metadata:"), evidence_message.index("snippet:"))

    def test_docs_only_option_request_requires_options_section(self) -> None:
        state = build_graph_state_input(
            user_input="matplotlib pie 차트 옵션을 정리해줘",
            messages=[HumanMessage(content="matplotlib pie 차트 옵션을 정리해줘")],
        )

        messages, _history_before, _history_after = build_synthesis_messages(
            state=state,
            action_rules=[],
            deduped_evidence=[
                {
                    "kind": "official",
                    "source_id": "url:https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html",
                    "url_or_path": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html",
                    "title": "matplotlib.pyplot.pie",
                    "snippet": "Plot a pie chart.",
                    "score": 0.95,
                    "doc_metadata": {
                        "symbol": "matplotlib.pyplot.pie",
                        "parameters": [{"name": "autopct", "description": "Controls numeric labels."}],
                    },
                }
            ],
            attempt=1,
            max_turns=6,
        )

        system_messages = [
            str(message.content) for message in messages if isinstance(message, SystemMessage)
        ]
        turn_message = next(content for content in system_messages if "[Turn Contract]" in content)
        self.assertIn("required_sections=options", turn_message)
        self.assertIn("options_section_required=true", turn_message)
        self.assertIn("candidate_facts or doc_metadata", turn_message)
        self.assertIn("options_answer_policy=answer first with confirmed items", turn_message)
        self.assertIn("do not replace the requested options summary", turn_message)
        self.assertIn("needs_more_evidence", turn_message)
        self.assertIn("wrapper/delegated API relationship", turn_message)

    def test_hybrid_presentation_request_preserves_source_comparison_layout(self) -> None:
        state = build_graph_state_input(
            user_input="Compare official docs with uploaded code as a summary and checklist.",
            messages=[HumanMessage(content="Compare official docs with uploaded code as a summary and checklist.")],
        )

        messages, _history_before, _history_after = build_synthesis_messages(
            state=state,
            action_rules=[],
            deduped_evidence=[
                {
                    "kind": "official",
                    "source_id": "url:https://docs.example.com",
                    "url_or_path": "https://docs.example.com",
                    "title": "Official Docs",
                    "snippet": "Official guidance.",
                    "score": 0.9,
                },
                {
                    "kind": "local",
                    "tool": "upload_search",
                    "source_id": "path:uploads/demo.py#chunk=0",
                    "url_or_path": "uploads/demo.py",
                    "snippet": "Uploaded code.",
                    "score": 0.8,
                },
            ],
            attempt=1,
            max_turns=6,
        )

        system_messages = [
            str(message.content) for message in messages if isinstance(message, SystemMessage)
        ]
        turn_message = next(content for content in system_messages if "[Turn Contract]" in content)
        self.assertIn("required_sections=summary, checklist, official_docs, upload_code, comparison", turn_message)
        self.assertIn("hybrid_layout=official_docs -> upload/local detail -> comparison", turn_message)
        self.assertIn("upload_code uses local/upload option_literals", turn_message)

    def test_code_example_request_requires_fenced_code_block_in_turn_contract(self) -> None:
        state = build_graph_state_input(
            user_input="BeautifulSoup으로 특정 태그 찾는 예제를 보여줘",
            messages=[HumanMessage(content="BeautifulSoup으로 특정 태그 찾는 예제를 보여줘")],
        )

        messages, _history_before, _history_after = build_synthesis_messages(
            state=state,
            action_rules=[],
            deduped_evidence=[
                {
                    "kind": "official",
                    "source_id": "url:https://www.crummy.com/software/BeautifulSoup/bs4/doc/",
                    "url_or_path": "https://www.crummy.com/software/BeautifulSoup/bs4/doc/",
                    "title": "Beautiful Soup Documentation",
                    "snippet": "Use find() and find_all() to search the parse tree.",
                    "score": 0.9,
                }
            ],
            attempt=1,
            max_turns=6,
        )

        system_messages = [
            str(message.content) for message in messages if isinstance(message, SystemMessage)
        ]
        turn_message = next(content for content in system_messages if "[Turn Contract]" in content)
        self.assertIn("required_sections=code_example", turn_message)
        self.assertIn("code_block_required=true", turn_message)
        self.assertIn("fenced code block", turn_message)
        self.assertIn("do not answer with prose only", turn_message)


if __name__ == "__main__":
    unittest.main()
