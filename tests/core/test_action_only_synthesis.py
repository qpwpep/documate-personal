import unittest

from langchain_core.messages import AIMessage, HumanMessage

from src.runtime.nodes.synthesis import make_synthesize_node

from .helpers import _CaptureSynthesizeLLM, build_legacy_state


def _response(result):
    return result["response"]


class ActionOnlySynthesisTest(unittest.TestCase):
    def test_action_only_save_without_previous_answer_reaches_llm(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="save this answer to txt")],
                    "user_input": "save this answer to txt",
                    "retrieved_evidence": [],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertIsNotNone(capture_llm.last_messages)
        self.assertEqual(_response(updates).final_answer, "synth result")

    def test_action_only_slack_with_previous_answer_reaches_llm_before_postprocess(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            build_legacy_state(
                {
                    "messages": [
                        HumanMessage(content="Explain numpy broadcasting."),
                        AIMessage(content="previous answer"),
                        HumanMessage(content="send this to slack"),
                    ],
                    "user_input": "send this to slack",
                    "retrieved_evidence": [],
                    "synthesis_attempt": 0,
                    "session_metadata": {
                        "slack_destination": {
                            "channel_id": "C123BENCH",
                            "user_id": None,
                            "email": None,
                        }
                    },
                }
            )
        )

        self.assertIsNotNone(capture_llm.last_messages)
        self.assertEqual(_response(updates).final_answer, "synth result")


if __name__ == "__main__":
    unittest.main()
