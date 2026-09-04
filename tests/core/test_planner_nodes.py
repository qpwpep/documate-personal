import unittest

from hypothesis import given, strategies as st
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
    @given(
        query=st.one_of(st.text(min_size=1, max_size=120), st.sampled_from([
            "official docs", "공식 문서는 제외해", "save this to txt", "the uploaded notebook",
        ])),
        routes=st.sampled_from([[], ["docs"], ["upload"], ["docs", "upload"], ["upload", "docs"]]),
        has_retriever=st.booleans(),
    )
    def test_valid_source_plan_survives_user_wording(self, query, routes, has_retriever) -> None:
        requested = PlannerOutput(use_retrieval=bool(routes), tasks=[
            RetrievalTask(route=route, query="멱등성 키 및 빈 문자열 비교", k=3) for route in routes
        ])
        result = make_planner_node(_CapturePlannerLLM(requested), verbose=False)(build_legacy_state({
            "user_input": query, "messages": [HumanMessage(content=query)],
            "retriever": object() if has_retriever else None,
        }))["planner"]
        missing_file = "upload" in routes and not has_retriever
        self.assertEqual(
            {"plan": result.output, "followup": bool(result.guided_followup)},
            {"plan": PlannerOutput.fallback() if missing_file else requested, "followup": missing_file},
        )

    @given(query=st.sampled_from([
        "이번 세션에 Python 결제 API 코드를 담은 .py 파일을 올려 뒀어. 그 코드에서 멱등성 키를 어떻게 처리하는지 찾아줘.",
        "Summarize the notebook (.ipynb) I uploaded. Exclude official documentation.",
        "공식 문서는 제외하고 그 코드에 적힌 내용만 확인해 주세요.",
    ]))
    def test_reviewed_upload_plan_does_not_gain_keyword_docs(self, query) -> None:
        requested = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="결제 코드 멱등성 키", k=4)])
        result = make_planner_node(_CapturePlannerLLM(requested), verbose=False)(build_legacy_state({
            "user_input": query, "messages": [HumanMessage(content=query)], "retriever": object(),
        }))["planner"]
        self.assertEqual(result.output, requested)

    @given(query=st.sampled_from([
        "공식 문서만으로 설명해줘", "내 파일에서 찾아줘", "검토한 뒤 저장해줘",
    ]))
    def test_planning_failure_requests_retry_without_guessing_sources(self, query) -> None:
        result = make_planner_node(_FailingPlannerLLM(), verbose=False)(build_legacy_state({
            "user_input": query, "messages": [HumanMessage(content=query)], "retriever": object(),
        }))["planner"]
        self.assertEqual(
            {"plan": result.output, "reason": result.diagnostics.reason, "followup": bool(result.guided_followup)},
            {"plan": PlannerOutput.fallback(), "reason": "planner_unavailable", "followup": True},
        )

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

    def test_planner_preserves_docs_search_from_model(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="FastAPI response_model", k=4)]))
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
        self.assertEqual(updates["planner"].output.tasks[0].query, "FastAPI response_model")

    def test_planner_preserves_upload_search_from_model(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="groupby usage", k=4)]))
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

    def test_planner_preserves_hybrid_search_from_model(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="pandas concat", k=4), RetrievalTask(route="upload", query="pandas concat uploaded notebook example", k=4)]))
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
        self.assertEqual(updates["planner"].output.tasks[0].query, "pandas concat")
        self.assertEqual(updates["planner"].output.tasks[1].query, "pandas concat uploaded notebook example")

    def test_upload_query_sanitizer_preserves_missing_identifier_tokens_for_hybrid_requests(self) -> None:
        sanitized = sanitize_retrieval_query(
            route="upload",
            query="train_test_split 공식 문법을 설명하고 업로드 노트북의 실제 사용 예를 찾아줘.",
        )
        self.assertIn("train_test_split", sanitized)
        self.assertIn("업로드", sanitized)

    def test_planner_blocks_upload_route_when_retriever_missing(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="groupby usage", k=4)]))
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

    def test_planner_requests_upload_when_local_notebook_is_not_available(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="dataframe joins example", k=4)]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Show a local notebook example for dataframe joins.")],
                    "user_input": "Show a local notebook example for dataframe joins.",
                }
            )
        )

        self.assertEqual(updates["planner"].output, PlannerOutput.fallback())
        self.assertEqual(updates["planner"].diagnostics.required_routes, ["upload"])
        self.assertIsNotNone(updates["planner"].guided_followup)


    @given(route=st.one_of(st.just("archive"), st.text().filter(lambda value: value not in {"docs", "upload", "local"})))
    def test_planner_schema_rejects_unsupported_retrieval_routes(self, route: str) -> None:
        with self.assertRaises(ValidationError):
            RetrievalTask(route=route, query="pandas merge", k=4)

    def test_planner_reports_failure_when_model_selects_unsupported_route(self) -> None:
        query = "Find pandas merge in my project code."
        planner = make_planner_node(_CapturePlannerLLM({
            "use_retrieval": True,
            "tasks": [{"route": "archive", "query": "pandas merge", "k": 4}],
        }), verbose=False)

        result = planner(build_legacy_state({
            "user_input": query,
            "messages": [HumanMessage(content=query)],
            "retriever": object(),
        }))

        self.assertEqual(result["planner"].output, PlannerOutput.fallback())
        self.assertEqual(result["planner"].diagnostics.reason, "planner_unavailable")
        self.assertTrue(result["planner"].guided_followup)
        self.assertIn("PLANNER_SCHEMA_INVALID", result["debug"].error_codes)


    def test_planner_uses_docs_when_no_file_source_is_requested(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="pandas merge examples", k=4)]))
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



    @given(has_retriever=st.booleans(), compare_docs=st.booleans())
    def test_planner_requests_missing_file_when_only_llm_recognizes_source(
        self, has_retriever: bool, compare_docs: bool,
    ) -> None:
        query = "Review the material I just sent."
        if compare_docs:
            query += " Compare with official docs."
        routes = ["docs", "upload"] if compare_docs else ["upload"]
        llm = _CapturePlannerLLM(PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route=route, query=query, k=4) for route in routes],
        ))

        result = make_planner_node(llm, verbose=False)(build_legacy_state({
            "user_input": query,
            "messages": [HumanMessage(content=query)],
            "retriever": object() if has_retriever else None,
        }))["planner"]

        self.assertEqual(
            {"routes": [task.route for task in result.output.tasks], "needs_file": bool(result.guided_followup)},
            {"routes": routes if has_retriever else [], "needs_file": not has_retriever},
        )


    @given(case=st.sampled_from([
        ("Explain Python file upload handling from official docs.", {"python", "upload"}),
        ("FastAPI 파일 업로드 API를 공식 문서 기준으로 설명해줘.", {"fastapi", "업로드"}),
        ("Explain the difference between .py and .ipynb files using official docs.", {".py", ".ipynb"}),
        ("업로드 파일은 사용하지 말고 pandas merge를 공식 문서만으로 설명해줘.", {"pandas", "merge"}),
    ]))
    def test_docs_query_preserves_subject_when_file_terms_are_topics(self, case: tuple) -> None:
        query, subjects = case
        sanitized = sanitize_retrieval_query(route="docs", query=query).lower()
        self.assertEqual({subject for subject in subjects if subject in sanitized}, subjects)


    def test_planner_falls_back_when_schema_invalid(self) -> None:
        planner_node = make_planner_node(_InvalidPlannerLLM(), verbose=False)
        updates = planner_node(build_legacy_state({"messages": [HumanMessage(content="hi")], "user_input": "hi"}))

        self.assertFalse(updates["planner"].output.use_retrieval)
        self.assertTrue(any("validation failed" in error for error in updates["debug"].planner_errors))

    def test_planner_preserves_no_retrieval_for_answer_delivery(self) -> None:
        capture_planner = _CapturePlannerLLM(
            PlannerOutput.fallback()
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
