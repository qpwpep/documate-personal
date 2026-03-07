import unittest

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.nodes.planner import make_planner_node
from src.planner_schema import PlannerOutput, RetrievalTask

from .helpers import _CapturePlannerLLM, _FailingPlannerLLM, _InvalidPlannerLLM


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

    def test_planner_node_falls_back_when_invoke_fails(self) -> None:
        planner_node = make_planner_node(_FailingPlannerLLM(), verbose=False)
        updates = planner_node({"messages": [HumanMessage(content="hi")], "user_input": "hi"})
        planner_output = updates["planner_output"]
        self.assertFalse(planner_output.use_retrieval)
        self.assertEqual(planner_output.tasks, [])
        self.assertTrue(any("planner" in error for error in updates.get("retrieval_errors", [])))

    def test_planner_node_uses_docs_heuristic_fallback_when_invoke_fails(self) -> None:
        planner_node = make_planner_node(_FailingPlannerLLM(), verbose=False)
        updates = planner_node(
            {
                "messages": [HumanMessage(content="공식 문서 기준으로 FastAPI response_model 설명해줘")],
                "user_input": "공식 문서 기준으로 FastAPI response_model 설명해줘",
            }
        )

        planner_output = updates["planner_output"]
        self.assertTrue(planner_output.use_retrieval)
        self.assertEqual([task.route for task in planner_output.tasks], ["docs"])
        self.assertEqual(updates["planner_status"], "heuristic_fallback")
        self.assertEqual(updates["planner_diagnostics"]["fallback_routes"], ["docs"])
        self.assertTrue(updates["planner_diagnostics"]["intent_required"])
        self.assertEqual(updates["planner_diagnostics"]["required_routes"], ["docs"])
        self.assertFalse(updates["planner_diagnostics"]["override_applied"])

    def test_planner_node_soft_overrides_valid_false_output_for_docs_query(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            {
                "messages": [HumanMessage(content="FastAPI response_model을 공식 문서 기준으로 설명해줘")],
                "user_input": "FastAPI response_model을 공식 문서 기준으로 설명해줘",
            }
        )

        planner_output = updates["planner_output"]
        self.assertTrue(planner_output.use_retrieval)
        self.assertEqual([task.route for task in planner_output.tasks], ["docs"])
        self.assertTrue(updates["planner_diagnostics"]["intent_required"])
        self.assertEqual(updates["planner_diagnostics"]["required_routes"], ["docs"])
        self.assertTrue(updates["planner_diagnostics"]["override_applied"])
        self.assertEqual(
            updates["planner_diagnostics"]["override_reason"],
            "missing_required_retrieval",
        )
        self.assertEqual(planner_output.tasks[0].query, "FastAPI response_model을 공식 문서 기준으로 설명해줘")

    def test_planner_node_falls_back_when_schema_invalid(self) -> None:
        planner_node = make_planner_node(_InvalidPlannerLLM(), verbose=False)
        updates = planner_node({"messages": [HumanMessage(content="hi")], "user_input": "hi"})
        planner_output = updates["planner_output"]
        self.assertFalse(planner_output.use_retrieval)
        self.assertEqual(planner_output.tasks, [])
        self.assertTrue(any("validation failed" in error for error in updates.get("retrieval_errors", [])))

    def test_planner_node_uses_upload_fallback_when_schema_invalid_and_retriever_available(self) -> None:
        planner_node = make_planner_node(_InvalidPlannerLLM(), verbose=False)
        updates = planner_node(
            {
                "messages": [HumanMessage(content="업로드한 파일에서 groupby를 찾아줘")],
                "user_input": "업로드한 파일에서 groupby를 찾아줘",
                "retriever": object(),
            }
        )

        planner_output = updates["planner_output"]
        self.assertTrue(planner_output.use_retrieval)
        self.assertEqual([task.route for task in planner_output.tasks], ["upload"])
        self.assertEqual(updates["planner_status"], "heuristic_fallback")

    def test_planner_node_sets_guided_followup_when_upload_requested_without_retriever(self) -> None:
        planner_node = make_planner_node(_InvalidPlannerLLM(), verbose=False)
        updates = planner_node(
            {
                "messages": [HumanMessage(content="업로드한 파일에서 groupby를 찾아줘")],
                "user_input": "업로드한 파일에서 groupby를 찾아줘",
            }
        )

        self.assertFalse(updates["planner_output"].use_retrieval)
        self.assertIn("업로드", updates["guided_followup"])
        self.assertTrue(updates["planner_diagnostics"]["intent_required"])
        self.assertEqual(updates["planner_diagnostics"]["required_routes"], ["upload"])
        self.assertTrue(updates["planner_diagnostics"]["override_applied"])
        self.assertEqual(
            updates["planner_diagnostics"]["override_reason"],
            "upload_retriever_missing",
        )

    def test_planner_node_soft_overrides_valid_false_output_for_upload_query(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            {
                "messages": [HumanMessage(content="업로드한 파일에서 groupby를 찾아줘")],
                "user_input": "업로드한 파일에서 groupby를 찾아줘",
                "retriever": object(),
            }
        )

        planner_output = updates["planner_output"]
        self.assertTrue(planner_output.use_retrieval)
        self.assertEqual([task.route for task in planner_output.tasks], ["upload"])
        self.assertTrue(updates["planner_diagnostics"]["intent_required"])
        self.assertEqual(updates["planner_diagnostics"]["required_routes"], ["upload"])
        self.assertTrue(updates["planner_diagnostics"]["override_applied"])
        self.assertEqual(
            updates["planner_diagnostics"]["override_reason"],
            "missing_required_retrieval",
        )

    def test_planner_node_adds_missing_hybrid_route_without_dropping_existing_route(self) -> None:
        capture_planner = _CapturePlannerLLM(
            PlannerOutput(
                use_retrieval=True,
                tasks=[RetrievalTask(route="docs", query="pandas concat official docs", k=3)],
            )
        )
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            {
                "messages": [HumanMessage(content="pandas concat 공식 문서 설명과 업로드 파일 예제를 함께 보여줘")],
                "user_input": "pandas concat 공식 문서 설명과 업로드 파일 예제를 함께 보여줘",
                "retriever": object(),
            }
        )

        planner_output = updates["planner_output"]
        self.assertTrue(planner_output.use_retrieval)
        self.assertEqual([task.route for task in planner_output.tasks], ["docs", "upload"])
        self.assertEqual(planner_output.tasks[0].query, "pandas concat official docs")
        self.assertEqual(planner_output.tasks[1].query, "pandas concat 공식 문서 설명과 업로드 파일 예제를 함께 보여줘")
        self.assertTrue(updates["planner_diagnostics"]["intent_required"])
        self.assertEqual(updates["planner_diagnostics"]["required_routes"], ["docs", "upload"])
        self.assertTrue(updates["planner_diagnostics"]["override_applied"])
        self.assertEqual(
            updates["planner_diagnostics"]["override_reason"],
            "missing_required_routes",
        )

    def test_planner_node_does_not_force_docs_for_upload_only_parameter_query(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            {
                "messages": [HumanMessage(content="sample_pipeline.ipynb 기준으로 train_test_split 파라미터를 찾아줘")],
                "user_input": "sample_pipeline.ipynb 기준으로 train_test_split 파라미터를 찾아줘",
                "retriever": object(),
            }
        )

        planner_output = updates["planner_output"]
        self.assertTrue(planner_output.use_retrieval)
        self.assertEqual([task.route for task in planner_output.tasks], ["upload"])
        self.assertEqual(updates["planner_diagnostics"]["required_routes"], ["upload"])

    def test_planner_skips_llm_for_action_only_request(self) -> None:
        capture_planner = _CapturePlannerLLM(
            PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="numpy", k=3)])
        )
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            {
                "messages": [HumanMessage(content="save this answer to txt")],
                "user_input": "save this answer to txt",
            }
        )

        self.assertEqual(capture_planner.call_count, 0)
        self.assertFalse(updates["planner_output"].use_retrieval)
        self.assertEqual(updates["planner_output"].tasks, [])

    def test_planner_includes_retry_context_system_message_on_retry(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        _ = planner_node(
            {
                "messages": [HumanMessage(content="hi")],
                "user_input": "hi",
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

        retry_context_messages = [
            message.content
            for message in (capture_planner.last_messages or [])
            if isinstance(message, SystemMessage) and "[Retry Context]" in str(message.content)
        ]
        self.assertEqual(len(retry_context_messages), 1)
        self.assertIn("reason=no_evidence", retry_context_messages[0])
        self.assertIn("previous_tasks=docs:numpy(k=3)", retry_context_messages[0])
