import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.nodes.synthesis import make_synthesize_node
from src.nodes.validation import make_validate_evidence_node
from src.planner_schema import PlannerOutput, RetrievalTask
from src.prompts import SYS_POLICY

from .helpers import (
    _CaptureStructuredSynthesizeLLM,
    _CaptureSynthesizeLLM,
    _TimeoutStructuredSynthesizeLLM,
)


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
                        "document_id": "url:https://numpy.org/doc/stable/",
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

    def test_validate_evidence_retries_when_claims_reference_unknown_evidence(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="local", query="example", k=3)],
        )
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [
                    {
                        "kind": "local",
                        "tool": "rag_search",
                        "source_id": "path:data/notebooks/example.ipynb#cell=0;chunk=0;start=0;end=12",
                        "document_id": "path:data/notebooks/example.ipynb",
                        "url_or_path": "data/notebooks/example.ipynb",
                        "snippet": "example snippet",
                        "score": 0.9,
                        "cell_id": 0,
                        "chunk_id": 0,
                        "start_offset": 0,
                        "end_offset": 12,
                    }
                ],
                "response_payload": {
                    "answer": "unsupported answer",
                    "claims": [
                        {
                            "text": "unsupported answer",
                            "evidence_ids": ["path:data/notebooks/example.ipynb#cell=0;chunk=99;start=0;end=12"],
                            "confidence": 0.6,
                        }
                    ],
                    "evidence": [],
                    "confidence": 0.6,
                },
                "retry_context": {
                    "attempt": 0,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                },
            }
        )
        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["retry_context"]["retry_reason"], "unsupported_claims")

    def test_validate_evidence_filters_to_valid_claims_after_retry_budget(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="local", query="example", k=3)],
        )
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [
                    {
                        "kind": "local",
                        "tool": "rag_search",
                        "source_id": "path:data/notebooks/example.ipynb#cell=0;chunk=0;start=0;end=12",
                        "document_id": "path:data/notebooks/example.ipynb",
                        "url_or_path": "data/notebooks/example.ipynb",
                        "snippet": "example snippet",
                        "score": 0.9,
                        "cell_id": 0,
                        "chunk_id": 0,
                        "start_offset": 0,
                        "end_offset": 12,
                    }
                ],
                "response_payload": {
                    "answer": "kept [1] dropped [2]",
                    "claims": [
                        {
                            "text": "kept",
                            "evidence_ids": ["path:data/notebooks/example.ipynb#cell=0;chunk=0;start=0;end=12"],
                            "confidence": 0.9,
                        },
                        {
                            "text": "dropped",
                            "evidence_ids": ["path:data/notebooks/example.ipynb#cell=0;chunk=99;start=0;end=12"],
                            "confidence": 0.1,
                        },
                    ],
                    "evidence": [],
                    "confidence": 0.5,
                },
                "retry_context": {
                    "attempt": 1,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                },
            }
        )
        self.assertFalse(result["needs_retry"])
        self.assertEqual(result["final_answer"], "kept [1]")
        self.assertEqual(len(result["response_payload"]["claims"]), 1)
        self.assertEqual(len(result["response_payload"]["evidence"]), 1)

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
        self.assertEqual(updates["response_payload"]["claims"], [])

    def test_synthesize_action_only_slack_requests_destination_without_metadata(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="send this to slack")],
                "user_input": "send this to slack",
                "retrieved_evidence": [],
                "synthesis_attempt": 0,
                "session_metadata": {"slack_destination": None},
            }
        )

        self.assertIsNone(capture_llm.last_messages)
        self.assertIn("channel_id", updates["final_answer"])
        self.assertEqual(updates["response_payload"]["claims"], [])

    def test_synthesize_action_only_slack_uses_session_metadata_without_followup(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            {
                "messages": [
                    HumanMessage(content="Explain numpy broadcasting."),
                    AIMessage(content="previous answer"),
                    HumanMessage(content="send this to slack"),
                ],
                "user_input": "send this to slack",
                "retrieved_evidence": [],
                "synthesis_attempt": 0,
                "session_metadata": {
                    "slack_destination": {
                        "channel_id": "C123BENCH",
                        "user_id": None,
                        "email": None,
                    }
                },
            }
        )

        self.assertIsNone(capture_llm.last_messages)
        self.assertEqual(updates["final_answer"], "previous answer")
        self.assertEqual(updates["response_payload"]["answer"], "previous answer")

    def test_synthesize_short_circuits_guided_followup(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="업로드한 파일에서 groupby를 찾아줘")],
                "user_input": "업로드한 파일에서 groupby를 찾아줘",
                "guided_followup": "업로드한 파일을 먼저 올려 주세요.",
                "retrieved_evidence": [],
                "synthesis_attempt": 0,
            }
        )

        self.assertIsNone(capture_llm.last_messages)
        self.assertEqual(updates["final_answer"], "업로드한 파일을 먼저 올려 주세요.")

    def test_synthesize_structures_claims_and_adopted_evidence(self) -> None:
        capture_llm = _CaptureStructuredSynthesizeLLM(
            {
                "answer": "Structured answer",
                "claims": [
                    {
                        "text": "Broadcasting expands compatible array shapes.",
                        "evidence_ids": ["url:https://numpy.org/doc/stable/"],
                        "confidence": 0.92,
                    }
                ],
                "confidence": 0.92,
            }
        )
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=8)

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                "retrieved_evidence": [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://numpy.org/doc/stable/",
                        "document_id": "url:https://numpy.org/doc/stable/",
                        "url_or_path": "https://numpy.org/doc/stable/",
                        "title": "NumPy Docs",
                        "snippet": "broadcasting rule",
                        "score": 0.9,
                    }
                ],
                "synthesis_attempt": 0,
            }
        )

        self.assertEqual(updates["final_answer"], "Broadcasting expands compatible array shapes. [1]")
        self.assertEqual(len(updates["response_payload"]["claims"]), 1)
        self.assertEqual(len(updates["response_payload"]["evidence"]), 1)
        self.assertEqual(
            updates["response_payload"]["claims"][0]["evidence_ids"],
            ["url:https://numpy.org/doc/stable/"],
        )

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
                        "document_id": "url:https://old.example.com/",
                        "url_or_path": "https://old.example.com/",
                        "title": "Old Evidence",
                        "snippet": "old snippet",
                        "score": 0.8,
                    },
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://new.example.com/",
                        "document_id": "url:https://new.example.com/",
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

    def test_synthesize_timeout_uses_grounded_fallback_without_plain_llm_retry(self) -> None:
        timeout_llm = _TimeoutStructuredSynthesizeLLM()
        synthesize_node = make_synthesize_node(timeout_llm, verbose=False, max_turns=8)

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                "planner_output": PlannerOutput(
                    use_retrieval=True,
                    tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
                ),
                "retrieved_evidence": [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://numpy.org/doc/stable/",
                        "document_id": "url:https://numpy.org/doc/stable/",
                        "url_or_path": "https://numpy.org/doc/stable/",
                        "title": "NumPy Docs",
                        "snippet": "Broadcasting expands compatible array shapes.",
                        "score": 0.9,
                    }
                ],
                "synthesis_attempt": 0,
            }
        )

        self.assertEqual(timeout_llm.call_count, 1)
        self.assertIn("Broadcasting expands compatible array shapes.", updates["final_answer"])
        self.assertIn("[1]", updates["final_answer"])
        self.assertEqual(updates["response_payload"]["claims"][0]["evidence_ids"], ["url:https://numpy.org/doc/stable/"])
        self.assertEqual(updates["synthesis_errors"][0].startswith("synthesize: structured output timed out"), True)
        synthesis_attempts = [
            item for item in updates["latency_trace"] if item.get("kind") == "synthesis_attempt"
        ]
        self.assertEqual(synthesis_attempts[0]["mode"], "timeout_grounded_fallback")


if __name__ == "__main__":
    unittest.main()
