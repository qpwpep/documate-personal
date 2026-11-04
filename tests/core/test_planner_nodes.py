import unittest

from langchain_core.messages import HumanMessage

from src.core.planner_schema import (
    InitialPlannerOutput,
    RetrievalTask,
)
from src.core.request_contracts import MissingInformation, WireRequestContract
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.synthesis import make_synthesize_node

from .helpers import _CapturePlannerLLM, build_test_state, mock_chat_model


class PlannerNodeTest(unittest.TestCase):
    def test_missing_upload_requests_a_file_without_starting_retrieval(self) -> None:
        query = "Find groupby in my notebook."
        task = RetrievalTask(route="upload", query="groupby usage", k=4)
        requested = InitialPlannerOutput(use_retrieval=True, tasks=[task],
            task_interpretations=[{"requirement_id": task.requirement_id, "scope": "target", "aspect_roles": []}],
            request_contract=WireRequestContract(slack_recipient={"state": "omitted"}))
        state = build_test_state({
            "user_input": query, "messages": [HumanMessage(content=query)], "retriever": None,
        })
        state.update(make_planner_node(_CapturePlannerLLM(requested), verbose=False)(state))
        result = state["planner"]

        self.assertEqual(result.output.tasks, [])
        self.assertFalse(result.output.use_retrieval)
        self.assertTrue(result.guided_followup)
        self.assertEqual(result.diagnostics.required_routes, ["upload"])
        synthesize = make_synthesize_node(mock_chat_model(lambda messages: self.fail("Clarification needs no model call")))
        state.update(synthesize(state))
        outcome = ResponseAssembler().assemble(response=state)

        self.assertEqual(outcome.status, "needs_input")
        self.assertEqual(outcome.missing_slots, ["upload"])
        self.assertIsNone(outcome.response)

    def test_subject_clarification_does_not_request_an_upload(self) -> None:
        query = "Please explain it."
        requested = InitialPlannerOutput(use_retrieval=False, tasks=[], request_contract=WireRequestContract(
            slack_recipient={"state": "omitted"},
            missing_info=(MissingInformation(
                slot="subject", reason="unclear", question="What should I explain?",
            ),),
        ))
        state = build_test_state({"user_input": query, "messages": [HumanMessage(content=query)]})
        state.update(make_planner_node(_CapturePlannerLLM(requested), verbose=False)(state))
        synthesize = make_synthesize_node(mock_chat_model(lambda messages: self.fail("Clarification needs no model call")))
        state.update(synthesize(state))
        outcome = ResponseAssembler().assemble(response=state)

        self.assertEqual(outcome.status, "needs_input")
        self.assertEqual(outcome.missing_slots, ["subject"])
        self.assertEqual(outcome.message, "What should I explain?")


if __name__ == "__main__":
    unittest.main()
