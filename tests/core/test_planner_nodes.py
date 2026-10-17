import unittest

from langchain_core.messages import HumanMessage

from src.core.planner_schema import (
    PlannerOutput,
    RetrievalTask,
)
from src.runtime.nodes.planner import make_planner_node

from .helpers import _CapturePlannerLLM, build_test_state


class PlannerNodeTest(unittest.TestCase):
    def test_missing_upload_requests_a_file_without_starting_retrieval(self) -> None:
        query = "Find groupby in my notebook."
        requested = PlannerOutput(use_retrieval=True, tasks=[
            RetrievalTask(route="upload", query="groupby usage", k=4),
        ])
        result = make_planner_node(_CapturePlannerLLM(requested), verbose=False)(build_test_state({
            "user_input": query, "messages": [HumanMessage(content=query)], "retriever": None,
        }))["planner"]

        self.assertEqual(result.output.tasks, [])
        self.assertFalse(result.output.use_retrieval)
        self.assertTrue(result.guided_followup)
        self.assertEqual(result.diagnostics.required_routes, ["upload"])


if __name__ == "__main__":
    unittest.main()
