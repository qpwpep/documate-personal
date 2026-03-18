import unittest

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.nodes.planner import make_planner_node
from src.planner_schema import PlannerOutput, RetrievalTask

from .helpers import _CapturePlannerLLM, _FailingPlannerLLM, _InvalidPlannerLLM, build_legacy_state


class PlannerNodeTest(unittest.TestCase):
    def test_planner_schema_rules(self) -> None:
        self.assertEqual(PlannerOutput(use_retrieval=False, tasks=[]).tasks, [])

        with self.assertRaises(ValidationError):
            PlannerOutput(
                use_retrieval=False,
                tasks=[RetrievalTask(route="docs", query="numpy", k=4)],
            )

        with self.assertRaises(ValidationError):
            PlannerOutput(use_retrieval=True, tasks=[])

        with self.assertRaises(ValidationError):
            PlannerOutput(
                use_retrieval=True,
                tasks=[
                    RetrievalTask(route="docs", query="numpy", k=4),
                    RetrievalTask(route="docs", query="python", k=4),
                ],
            )

    def test_planner_deterministically_routes_explicit_docs_request(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Explain FastAPI response_model from official docs.")],
                    "user_input": "Explain FastAPI response_model from official docs.",
                }
            )
        )

        self.assertEqual(capture_planner.call_count, 0)
        self.assertEqual(updates["planner"].status, "deterministic")
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["docs"])
        self.assertEqual(updates["planner"].output.tasks[0].query, "FastAPI response_model")

    def test_planner_deterministically_routes_upload_request(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Find groupby usage in the uploaded notebook.")],
                    "user_input": "Find groupby usage in the uploaded notebook.",
                    "retriever": object(),
                }
            )
        )

        self.assertEqual(capture_planner.call_count, 0)
        self.assertEqual(updates["planner"].status, "deterministic")
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["upload"])

    def test_planner_deterministically_routes_hybrid_docs_and_upload_request(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Explain pandas concat from official docs and compare it with the uploaded notebook example.")],
                    "user_input": "Explain pandas concat from official docs and compare it with the uploaded notebook example.",
                    "retriever": object(),
                }
            )
        )

        self.assertEqual(capture_planner.call_count, 0)
        self.assertEqual(updates["planner"].status, "deterministic")
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["docs", "upload"])
        self.assertEqual(updates["planner"].output.tasks[0].query, "pandas concat")
        self.assertEqual(updates["planner"].output.tasks[1].query, "uploaded notebook example")

    def test_planner_blocks_upload_route_when_retriever_missing(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Find groupby usage in the uploaded notebook.")],
                    "user_input": "Find groupby usage in the uploaded notebook.",
                }
            )
        )

        self.assertFalse(updates["planner"].output.use_retrieval)
        self.assertEqual(updates["planner"].status, "deterministic")
        self.assertEqual(updates["planner"].diagnostics.override_reason, "upload_retriever_missing")
        self.assertIsNotNone(updates["planner"].guided_followup)

    def test_planner_uses_local_route_only_for_explicit_local_intent(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Show a local notebook example for dataframe joins.")],
                    "user_input": "Show a local notebook example for dataframe joins.",
                }
            )
        )

        self.assertEqual(capture_planner.call_count, 0)
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["local"])

    def test_planner_does_not_open_local_route_when_docs_intent_is_explicit(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Explain pandas merge from official docs with an example.")],
                    "user_input": "Explain pandas merge from official docs with an example.",
                }
            )
        )

        self.assertEqual(capture_planner.call_count, 0)
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["docs"])

    def test_planner_uses_heuristic_fallback_when_llm_fails_on_non_deterministic_docs_like_query(self) -> None:
        planner_node = make_planner_node(_FailingPlannerLLM(), verbose=False)
        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="pandas merge parameter")],
                    "user_input": "pandas merge parameter",
                }
            )
        )

        self.assertEqual(updates["planner"].status, "heuristic_fallback")
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["docs"])
        self.assertTrue(any("planner" in error for error in updates.get("debug", {}).get("planner_errors", [])))

    def test_planner_falls_back_when_schema_invalid(self) -> None:
        planner_node = make_planner_node(_InvalidPlannerLLM(), verbose=False)
        updates = planner_node(build_legacy_state({"messages": [HumanMessage(content="hi")], "user_input": "hi"}))

        self.assertFalse(updates["planner"].output.use_retrieval)
        self.assertTrue(any("validation failed" in error for error in updates["debug"].planner_errors))

    def test_planner_skips_llm_for_action_only_request(self) -> None:
        capture_planner = _CapturePlannerLLM(
            PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="numpy", k=3)])
        )
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="save this answer to txt")],
                    "user_input": "save this answer to txt",
                }
            )
        )

        self.assertEqual(capture_planner.call_count, 0)
        self.assertFalse(updates["planner"].output.use_retrieval)

    def test_planner_records_llm_call_metadata_when_llm_path_is_used(self) -> None:
        capture_planner = _CapturePlannerLLM(
            PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="numpy", k=3)]),
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
        self.assertEqual(len(updates["debug"].llm_calls), 1)
        self.assertEqual(updates["debug"].llm_calls[0].stage, "planner")
        self.assertEqual(updates["debug"].llm_calls[0].path, "structured")

    def test_planner_includes_retry_context_system_message_on_retry(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        _ = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="numpy parameters")],
                    "user_input": "numpy parameters",
                    "planner_output": PlannerOutput(
                        use_retrieval=True,
                        tasks=[RetrievalTask(route="docs", query="numpy", k=3)],
                    ),
                    "retry_context": {
                        "attempt": 1,
                        "max_retries": 1,
                        "retry_reason": "no_evidence",
                        "retrieval_feedback": "query too narrow",
                    },
                }
            )
        )

        retry_context_messages = [
            message.content
            for message in (capture_planner.last_messages or [])
            if isinstance(message, SystemMessage) and "[Retry Context]" in str(message.content)
        ]
        self.assertEqual(len(retry_context_messages), 1)
        self.assertIn("reason=no_evidence", retry_context_messages[0])
        self.assertIn("previous_routes=docs", retry_context_messages[0])


if __name__ == "__main__":
    unittest.main()
