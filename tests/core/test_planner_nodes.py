import unittest

from hypothesis import given, strategies as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from src.core.contracts import PlannerDiagnostic, SessionMetadata, SlackDestination
from src.core.planner_schema import (
    PlannerOutput,
    RetrievalTask,
)
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.planner.query_sanitizer import sanitize_retrieval_query

from .helpers import (
    _CapturePlannerLLM,
    _FailingPlannerLLM,
    _InvalidPlannerLLM,
    build_test_state,
)


class PlannerNodeTest(unittest.TestCase):

    @given(query=st.sampled_from([
        "이번 세션에 Python 결제 API 코드를 담은 .py 파일을 올려 뒀어. 그 코드에서 멱등성 키를 어떻게 처리하는지 찾아줘.",
        "Summarize the notebook (.ipynb) I uploaded. Exclude official documentation.",
        "공식 문서는 제외하고 그 코드에 적힌 내용만 확인해 주세요.",
    ]))
    def test_reviewed_upload_plan_does_not_gain_keyword_docs(self, query) -> None:
        requested = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="결제 코드 멱등성 키", k=4)])
        result = make_planner_node(_CapturePlannerLLM(requested), verbose=False)(build_test_state({
            "user_input": query, "messages": [HumanMessage(content=query)], "retriever": object(),
        }))["planner"]
        self.assertEqual(result.output, requested)

    @given(query=st.sampled_from([
        "공식 문서만으로 설명해줘", "내 파일에서 찾아줘", "검토한 뒤 저장해줘",
    ]))
    def test_planning_failure_requests_retry_without_guessing_sources(self, query) -> None:
        result = make_planner_node(_FailingPlannerLLM(), verbose=False)(build_test_state({
            "user_input": query, "messages": [HumanMessage(content=query)], "retriever": object(),
        }))["planner"]
        self.assertEqual(
            {"plan": result.output, "reason": result.diagnostics.reason, "followup": bool(result.guided_followup)},
            {"plan": PlannerOutput.fallback(), "reason": "planner_unavailable", "followup": True},
        )



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
            build_test_state(
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
            build_test_state(
                {
                    "messages": [HumanMessage(content="Show a local notebook example for dataframe joins.")],
                    "user_input": "Show a local notebook example for dataframe joins.",
                }
            )
        )

        self.assertEqual(updates["planner"].output, PlannerOutput.fallback())
        self.assertEqual(updates["planner"].diagnostics.required_routes, ["upload"])
        self.assertIsNotNone(updates["planner"].guided_followup)





    def test_planner_uses_docs_when_no_file_source_is_requested(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="pandas merge examples", k=4)]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        updates = planner_node(
            build_test_state(
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

        result = make_planner_node(llm, verbose=False)(build_test_state({
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
        updates = planner_node(build_test_state({"messages": [HumanMessage(content="hi")], "user_input": "hi"}))

        self.assertFalse(updates["planner"].output.use_retrieval)
        self.assertTrue(any("validation failed" in error for error in updates["debug"].planner_errors))

    def test_planner_includes_retry_context_system_message_on_retry(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        _ = planner_node(
            build_test_state(
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
