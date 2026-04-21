import unittest

from langchain_core.messages import HumanMessage

from src.core.contracts.boundary.planner import parse_planner_output
from src.runtime.nodes.planner import make_planner_node
from src.core.planner_schema import PlannerOutput

from .helpers import _CapturePlannerLLM, build_legacy_state


class PlannerSchemaMismatchTest(unittest.TestCase):
    def test_parse_planner_output_accepts_planner_model_instance(self) -> None:
        errors: list[str] = []
        result = parse_planner_output(
            PlannerOutput(
                use_retrieval=True,
                tasks=[{"route": "docs", "query": "numpy", "k": 3}],
            ),
            errors,
        )

        self.assertEqual(errors, [])
        self.assertTrue(result.use_retrieval)
        self.assertEqual([task.route for task in result.tasks], ["docs"])

    def test_planner_node_accepts_planner_schema_from_structured_wrapper(self) -> None:
        capture_planner = _CapturePlannerLLM(
            PlannerOutput(
                use_retrieval=True,
                tasks=[{"route": "docs", "query": "numpy parameters", "k": 3}],
            ),
            include_raw=True,
        )
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="numpy parameters")],
                    "user_input": "numpy parameters",
                }
            )
        )

        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(updates["debug"].planner_errors, [])
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["docs"])
        self.assertEqual(updates["planner"].output.tasks[0].query, "numpy parameters")


if __name__ == "__main__":
    unittest.main()
