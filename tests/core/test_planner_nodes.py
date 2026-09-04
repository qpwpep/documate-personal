import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from src.core.contracts import PlannerDiagnostic, SessionMetadata, SlackDestination
from src.core.planner_schema import (
    PLANNER_WARNING_DUPLICATE_ROUTE_MERGED,
    PlannerOutput,
    RetrievalTask,
)
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.planner.query_sanitizer import sanitize_retrieval_query

from .helpers import (
    _CapturePlannerLLM,
    _FailingPlannerLLM,
    _InvalidPlannerLLM,
    build_legacy_state,
)


class PlannerNodeTest(unittest.TestCase):
    def test_nested_state_models_are_not_subscriptable(self) -> None:
        with self.assertRaises(TypeError):
            _ = PlannerDiagnostic()["reason"]

        with self.assertRaises(TypeError):
            _ = SessionMetadata(slack_destination=SlackDestination(channel_id="C123"))["slack_destination"]

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

    def test_planner_uses_llm_and_guardrail_for_explicit_docs_request(self) -> None:
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

        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(updates["planner"].status, "llm")
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["docs"])
        self.assertEqual(updates["planner"].output.tasks[0].query, "Explain FastAPI response_model from official docs.")

    def test_planner_uses_llm_and_guardrail_for_upload_request(self) -> None:
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

        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(updates["planner"].status, "llm")
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["upload"])
        self.assertIn("groupby", updates["planner"].output.tasks[0].query.lower())

    def test_planner_uses_llm_and_guardrail_for_hybrid_docs_and_upload_request(self) -> None:
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

        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(updates["planner"].status, "llm")
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["docs", "upload"])
        self.assertEqual(updates["planner"].output.tasks[0].query, "Explain pandas concat from official docs and compare it with the uploaded notebook example.")
        self.assertEqual(updates["planner"].output.tasks[1].query, "Explain pandas concat from official docs and compare it with the uploaded notebook example.")

    def test_upload_query_sanitizer_preserves_missing_identifier_tokens_for_hybrid_requests(self) -> None:
        sanitized = sanitize_retrieval_query(
            route="upload",
            query="train_test_split 공식 문법을 설명하고 업로드 노트북의 실제 사용 예를 찾아줘.",
        )
        self.assertIn("train_test_split", sanitized)
        self.assertIn("업로드", sanitized)

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
        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(updates["planner"].status, "llm")
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

        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(updates["planner"].status, "llm")
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["local"])

    def test_planner_does_not_treat_generic_example_as_local_rag_intent(self) -> None:
        planner_node = make_planner_node(_FailingPlannerLLM(), verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="BeautifulSoup으로 특정 태그 찾는 예제를 보여줘")],
                    "user_input": "BeautifulSoup으로 특정 태그 찾는 예제를 보여줘",
                }
            )
        )

        self.assertEqual(updates["planner"].status, "heuristic_fallback")
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["docs"])
        self.assertNotIn("local", [task.route for task in updates["planner"].output.tasks])

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

        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(updates["planner"].status, "llm")
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
        self.assertTrue(any("planner" in error for error in updates["debug"].planner_errors))

    def test_planner_falls_back_when_schema_invalid(self) -> None:
        planner_node = make_planner_node(_InvalidPlannerLLM(), verbose=False)
        updates = planner_node(build_legacy_state({"messages": [HumanMessage(content="hi")], "user_input": "hi"}))

        self.assertFalse(updates["planner"].output.use_retrieval)
        self.assertTrue(any("validation failed" in error for error in updates["debug"].planner_errors))

    def test_planner_uses_llm_but_blocks_retrieval_for_action_only_request(self) -> None:
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

        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(updates["planner"].status, "llm")
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

    def test_planner_uses_one_structured_request(self) -> None:
        capture_planner = _CapturePlannerLLM(
            PlannerOutput(
                use_retrieval=True,
                tasks=[RetrievalTask(route="docs", query="numpy parameters", k=3)],
            ),
            include_raw=True,
        )
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Explain numpy parameters from official docs.")],
                    "user_input": "Explain numpy parameters from official docs.",
                }
            )
        )

        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(updates["planner"].status, "llm")
        self.assertEqual(updates["planner"].output.tasks[0].query, "numpy parameters")
        self.assertEqual([item.path for item in updates["debug"].llm_calls], ["structured"])

    def test_planner_prompt_preserves_library_name_for_docs_queries(self) -> None:
        capture_planner = _CapturePlannerLLM(
            PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="Bare", k=3)])
        )
        planner_node = make_planner_node(capture_planner, verbose=False)

        _ = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="bare parameters")],
                    "user_input": "bare parameters",
                }
            )
        )

        system_prompts = [
            str(message.content)
            for message in (capture_planner.last_messages or [])
            if isinstance(message, SystemMessage)
        ]
        self.assertTrue(
            any("preserve the library/framework name in task.query" in prompt for prompt in system_prompts)
        )

    def test_planner_accepts_structured_output_model_instances(self) -> None:
        capture_planner = _CapturePlannerLLM(
            PlannerOutput(
                use_retrieval=True,
                tasks=[{"route": "docs", "query": "numpy", "k": 3}],
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

        self.assertEqual(updates["planner"].status, "llm")
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["docs"])
        self.assertEqual(updates["planner"].output.tasks[0].query, "numpy")
        self.assertEqual(updates["debug"].planner_errors, [])

    def test_planner_merges_duplicate_routes_from_raw_structured_output(self) -> None:
        raw_payload = {
            "use_retrieval": True,
            "tasks": [
                {"route": "docs", "query": "numpy", "k": 3},
                {"route": "docs", "query": "pandas", "k": 5},
            ],
        }
        capture_planner = _CapturePlannerLLM(
            None,
            include_raw=True,
            raw_message=AIMessage(content="", additional_kwargs={"parsed": raw_payload}),
            parsing_error=ValueError("duplicate routes are not allowed in planner tasks"),
        )
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Compare numpy and pandas docs.")],
                    "user_input": "Compare numpy and pandas docs.",
                }
            )
        )

        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(updates["planner"].status, "llm")
        self.assertEqual(updates["debug"].planner_errors, [])
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["docs"])
        self.assertIn("numpy", updates["planner"].output.tasks[0].query)
        self.assertIn("pandas", updates["planner"].output.tasks[0].query)
        self.assertEqual(
            updates["planner"].diagnostics.planner_warnings,
            [PLANNER_WARNING_DUPLICATE_ROUTE_MERGED],
        )

    def test_planner_merges_duplicate_routes_from_parsed_payload_without_langchain_validation(self) -> None:
        raw_payload = {
            "use_retrieval": True,
            "tasks": [
                {"route": "docs", "query": "numpy", "k": 3},
                {"route": "docs", "query": "pandas", "k": 5},
            ],
        }
        capture_planner = _CapturePlannerLLM(
            raw_payload,
            include_raw=True,
        )
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Compare numpy and pandas docs.")],
                    "user_input": "Compare numpy and pandas docs.",
                }
            )
        )

        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(updates["planner"].status, "llm")
        self.assertEqual([task.route for task in updates["planner"].output.tasks], ["docs"])
        self.assertIn("numpy", updates["planner"].output.tasks[0].query)
        self.assertIn("pandas", updates["planner"].output.tasks[0].query)
        self.assertEqual(
            updates["planner"].diagnostics.planner_warnings,
            [PLANNER_WARNING_DUPLICATE_ROUTE_MERGED],
        )

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
