import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from src.nodes.synthesis import make_synthesize_node
from src.nodes.validation import make_validate_evidence_node
from src.planner_schema import PlannerOutput, RetrievalTask
from src.prompts import SYS_POLICY

from .helpers import _CaptureSynthesizeLLM


class SynthesisValidationTest(unittest.TestCase):
    def test_validate_evidence_retries_once_then_returns_route_specific_followup(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
        )

        first = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "retry_context": {
                    "attempt": 0,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                },
            }
        )
        second = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "retry_context": first["retry_context"],
            }
        )
        self.assertTrue(first["needs_retry"])
        self.assertEqual(first["retry_context"]["attempt"], 1)
        self.assertEqual(first["retry_context"]["retry_reason"], "no_evidence")
        self.assertFalse(second["needs_retry"])
        self.assertIn("final_answer", second)
        self.assertTrue(second["final_answer"].startswith("현재 확인 가능한 근거를 찾지 못했습니다."))
        self.assertIn("공식 문서", second["final_answer"])

    def test_validate_evidence_sets_tool_error_reason(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
        )
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "retrieval_errors": ["tavily_search: failed (timeout)"],
                "retry_context": {
                    "attempt": 0,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                },
            }
        )
        self.assertFalse(result["needs_retry"])
        self.assertEqual(result["retry_context"]["attempt"], 0)
        self.assertEqual(result["retry_context"]["retry_reason"], "tool_error")
        self.assertIn("final_answer", result)
        self.assertTrue(result["final_answer"].startswith("현재 확인 가능한 근거를 찾지 못했습니다."))
        self.assertIn("공식 문서", result["final_answer"])

    def test_validate_evidence_maps_upload_unavailable_to_blocked_missing_upload(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="upload", query="groupby", k=3)],
        )
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "retrieval_diagnostics": [
                    {
                        "tool": "upload_search",
                        "route": "upload",
                        "status": "unavailable",
                        "message": "upload retriever is unavailable",
                        "query": "groupby",
                        "attempt": 1,
                    }
                ],
                "retry_context": {
                    "attempt": 0,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                    "retrieval_diagnostic_start_index": 0,
                },
            }
        )
        self.assertFalse(result["needs_retry"])
        self.assertEqual(result["retry_context"]["retry_reason"], "blocked_missing_upload")
        self.assertIn("업로드", result["final_answer"])

    def test_validate_evidence_sets_low_score_reason(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
        )
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://numpy.org/doc/stable/",
                        "url_or_path": "https://numpy.org/doc/stable/",
                        "title": "NumPy Docs",
                        "snippet": "official docs",
                        "score": 0.2,
                    }
                ],
                "retry_context": {
                    "attempt": 0,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                },
            }
        )
        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["retry_context"]["retry_reason"], "low_score")
        self.assertAlmostEqual(result["retry_context"]["score_avg"], 0.2)

    def test_validate_evidence_passes_when_retrieval_not_required(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=False, tasks=[])
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "synthesis_attempt": 1,
            }
        )
        self.assertFalse(result["needs_retry"])

    def test_synthesize_node_keeps_sys_policy_persona(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                "retrieved_evidence": [],
                "synthesis_attempt": 0,
            }
        )
        self.assertEqual(updates["final_answer"], "synth result")
        self.assertIsNotNone(capture_llm.last_messages)
        self.assertIsInstance(capture_llm.last_messages[0], SystemMessage)
        self.assertEqual(capture_llm.last_messages[0].content, SYS_POLICY)

    def test_synthesize_short_circuits_action_only_save_request(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="save this answer to txt")],
                "user_input": "save this answer to txt",
                "retrieved_evidence": [],
                "synthesis_attempt": 0,
            }
        )

        self.assertIsNone(capture_llm.last_messages)
        self.assertEqual(updates["final_answer"], "요청하신 최종 답변을 저장합니다.")

    def test_synthesize_short_circuits_guided_followup(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="업로드한 파일에서 groupby를 찾아줘")],
                "user_input": "업로드한 파일에서 groupby를 찾아줘",
                "guided_followup": "업로드한 파일을 먼저 올려주세요.",
                "retrieved_evidence": [],
                "synthesis_attempt": 0,
            }
        )

        self.assertIsNone(capture_llm.last_messages)
        self.assertEqual(updates["final_answer"], "업로드한 파일을 먼저 올려주세요.")

    def test_synthesize_uses_only_current_attempt_evidence_window(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=8)

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                "retrieved_evidence": [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://old.example.com/",
                        "url_or_path": "https://old.example.com/",
                        "title": "Old Evidence",
                        "snippet": "old snippet",
                        "score": 0.8,
                    },
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://new.example.com/",
                        "url_or_path": "https://new.example.com/",
                        "title": "New Evidence",
                        "snippet": "new snippet",
                        "score": 0.9,
                    },
                ],
                "retry_context": {"evidence_start_index": 1},
                "synthesis_attempt": 0,
            }
        )

        self.assertEqual(updates["final_answer"], "synth result")
        retrieved_evidence_messages = [
            message.content
            for message in (capture_llm.last_messages or [])
            if isinstance(message, SystemMessage) and "[Retrieved Evidence]" in str(message.content)
        ]
        self.assertEqual(len(retrieved_evidence_messages), 1)
        self.assertIn("https://new.example.com/", retrieved_evidence_messages[0])
        self.assertNotIn("https://old.example.com/", retrieved_evidence_messages[0])
