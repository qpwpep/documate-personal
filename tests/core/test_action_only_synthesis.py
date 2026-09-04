import unittest

from langchain_core.messages import AIMessage, HumanMessage

from src.runtime.nodes.synthesis import make_synthesize_node
from src.core.contracts.graph_state import DebugState
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.runtime.nodes.synthesis.models import SynthesisContext
from src.runtime.nodes.synthesis.short_circuit import maybe_short_circuit_synthesis

from .helpers import _CaptureSynthesizeLLM, build_legacy_state


def _response(result):
    return result["response"]


class ActionOnlySynthesisTest(unittest.TestCase):
    def test_retrieval_plan_reaches_synthesis_despite_action_wording(self) -> None:
        context = SynthesisContext(
            attempt=1, user_input="그 항목을 확인해서 슬랙으로 보내줘", messages=[],
            guided_followup="", slack_target_available=False, parse_errors=[], planner_parse_errors=[],
            planner_output=PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="항목 확인", k=4)]),
            retrieval_required=True, primary_evidence_items=[], grounded_fallback_evidence_items=[],
        )
        result = maybe_short_circuit_synthesis(state={}, debug=DebugState(), context=context, stage_started=0.0)
        self.assertIsNone(result)

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
