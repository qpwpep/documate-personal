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
        instruction_messages = [
            content for content in system_messages if "[Synthesis Instructions]" in content
        ]
        self.assertEqual(len(instruction_messages), 1)
        self.assertTrue(all("[Action Request]" not in content for content in system_messages))
        self.assertEqual(sum(1 for content in system_messages if "[Retrieved Evidence]" in content), 1)
        self.assertTrue(all("[Official Docs Evidence]" not in content for content in system_messages))
        self.assertTrue(all("[Uploaded Code Evidence]" not in content for content in system_messages))
        self.assertIn("Action requests:", instruction_messages[0])
        self.assertIn("[Hybrid Synthesis]", instruction_messages[0])
        self.assertIn("official takeaway first", instruction_messages[0])
        self.assertIn("must cite only official docs source_id values", instruction_messages[0])
        self.assertIn("must cite only uploaded/local source_id values", instruction_messages[0])
        self.assertIn("Do not collapse the whole answer into only docs or only uploaded/local evidence", instruction_messages[0])
        self.assertIn("actual message body to save/share now", instruction_messages[0])
        self.assertIn("Do not answer with a checklist about the action itself", instruction_messages[0])
        self.assertIn("Ignore markdown formatting, breadcrumbs, navigation labels", instruction_messages[0])
        self.assertIn("Retry after evidence validation failed", instruction_messages[0])


if __name__ == "__main__":
    unittest.main()
