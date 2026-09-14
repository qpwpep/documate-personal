import unittest

from src.core.contracts.boundary.planner import parse_planner_output, parse_planner_state
from src.core.planner_schema import PlannerOutput


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

    def test_parse_planner_output_preserves_independent_source_requirements(self) -> None:
        errors: list[str] = []
        warnings: list[str] = []
        result = parse_planner_output(
            {
                "use_retrieval": True,
                "tasks": [
                    {"route": "docs", "query": "numpy", "k": 3},
                    {"route": "docs", "query": "pandas", "k": 5},
                ],
            },
            errors,
            warnings,
        )

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertTrue(result.use_retrieval)
        self.assertEqual([(task.route, task.query, task.k) for task in result.tasks],
                         [("docs", "numpy", 3), ("docs", "pandas", 5)])

    def test_parse_planner_state_keeps_two_sources_without_merge_warning(self) -> None:
        state = parse_planner_state(
            {
                "status": "llm",
                "output": {
                    "use_retrieval": True,
                    "tasks": [
                        {"route": "docs", "query": "numpy", "k": 3},
                        {"route": "docs", "query": "pandas", "k": 5},
                    ],
                },
                "diagnostics": {
                    "status": "llm",
                    "reason": None,
                    "fallback_routes": [],
                    "intent_required": True,
                    "required_routes": ["docs"],
                    "override_applied": False,
                    "override_reason": None,
                },
            }
        )

        self.assertEqual([task.route for task in state.output.tasks], ["docs", "docs"])
        self.assertEqual(state.diagnostics.planner_warnings, [])


if __name__ == "__main__":
    unittest.main()
