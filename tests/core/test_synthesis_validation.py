import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.nodes.session import make_summarize_node
from src.nodes.synthesis import make_synthesize_node
from src.nodes.validation import make_validate_evidence_node
from src.planner_schema import PlannerOutput, RetrievalTask
from src.prompts import SYS_POLICY

from .helpers import (
    _CaptureSummaryLLM,
    _CaptureStructuredSynthesizeLLM,
    _CaptureSynthesizeLLM,
    _StructuredThenPlainFallbackSynthesizeLLM,
    _TimeoutStructuredSynthesizeLLM,
)


class SynthesisValidationTest(unittest.TestCase):
    def test_validate_evidence_retries_once_for_docs_only_no_evidence(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
        )

        result = validate_node(
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

        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["retry_context"]["attempt"], 1)
        self.assertEqual(result["retry_context"]["retry_reason"], "no_evidence")
        self.assertNotIn("final_answer", result)

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

        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["retry_context"]["attempt"], 1)
        self.assertEqual(result["retry_context"]["retry_reason"], "tool_error")
        self.assertNotIn("final_answer", result)

    def test_validate_evidence_does_not_treat_planner_errors_as_tool_errors(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
        )
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "planner_errors": ["planner: structured output invocation failed (boom)"],
                "retry_context": {
                    "attempt": 0,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                    "retrieval_diagnostic_start_index": 0,
                },
            }
        )

        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["retry_context"]["retry_reason"], "no_evidence")

    def test_validate_evidence_retries_docs_only_tool_error_even_with_grounded_payload(self) -> None:
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
                        "snippet": "Broadcasting expands compatible array shapes.",
                        "score": 0.9,
                    }
                ],
                "retrieval_errors": ["tavily_search: failed (timeout)"],
                "response_payload": {
                    "answer": "draft",
                    "claims": [],
                    "evidence": [],
                    "confidence": None,
                },
                "retry_context": {
                    "attempt": 0,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                    "retrieval_diagnostic_start_index": 0,
                },
            }
        )

        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["retry_context"]["attempt"], 1)
        self.assertEqual(result["retry_context"]["retry_reason"], "tool_error")
        self.assertNotIn("final_answer", result)

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
        self.assertIn("final_answer", result)

    def test_validate_evidence_retries_docs_only_low_score(self) -> None:
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
                "response_payload": {
                    "answer": "draft",
                    "claims": [],
                    "evidence": [],
                    "confidence": None,
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
        self.assertEqual(result["retry_context"]["attempt"], 1)
        self.assertEqual(result["retry_context"]["retry_reason"], "low_score")
        self.assertAlmostEqual(result["retry_context"]["score_avg"], 0.2)
        self.assertNotIn("response_payload", result)

    def test_validate_evidence_salvages_upload_low_score_without_retry(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="upload", query="groupby", k=3)],
        )
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [
                    {
                        "kind": "local",
                        "tool": "upload_search",
                        "source_id": "path:uploads/demo/sample.py#chunk=0;start=0;end=48",
                        "document_id": "path:uploads/demo/sample.py",
                        "url_or_path": "uploads/demo/sample.py",
                        "snippet": 'grouped = all_sales.groupby("region")["amount"].sum()',
                        "score": 0.0,
                        "chunk_id": 0,
                        "start_offset": 0,
                        "end_offset": 48,
                    }
                ],
                "response_payload": {
                    "answer": "draft",
                    "claims": [],
                    "evidence": [],
                    "confidence": None,
                },
                "retry_context": {
                    "attempt": 0,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                },
            }
        )

        self.assertFalse(result["needs_retry"])
        self.assertEqual(result["retry_context"]["retry_reason"], "unsupported_claims")
        self.assertIn("groupby", result["final_answer"])

    def test_validate_evidence_retries_docs_half_of_docs_upload_and_preserves_upload_context(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[
                RetrievalTask(route="docs", query="train_test_split 공식 문법", k=3),
                RetrievalTask(route="upload", query="업로드 노트북 실제 사용 예", k=3),
            ],
        )
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://huggingface.co/docs/bad",
                        "document_id": "url:https://huggingface.co/docs/bad",
                        "url_or_path": "https://huggingface.co/docs/bad",
                        "snippet": "unrelated content",
                        "score": 0.1,
                    },
                    {
                        "kind": "local",
                        "tool": "upload_search",
                        "source_id": "path:uploads/demo/sample.ipynb#cell=1;chunk=0;start=0;end=64",
                        "document_id": "path:uploads/demo/sample.ipynb",
                        "url_or_path": "uploads/demo/sample.ipynb",
                        "snippet": "X_train, X_test, y_train, y_test = train_test_split(...)",
                        "score": 0.0,
                        "cell_id": 1,
                        "chunk_id": 0,
                        "start_offset": 0,
                        "end_offset": 64,
                    },
                ],
                "retrieval_diagnostics": [
                    {
                        "tool": "tavily_search",
                        "route": "docs",
                        "status": "success",
                        "message": "",
                        "query": "train_test_split 공식 문법",
                        "attempt": 1,
                    },
                    {
                        "tool": "upload_search",
                        "route": "upload",
                        "status": "success",
                        "message": "",
                        "query": "업로드 노트북 실제 사용 예",
                        "attempt": 1,
                    },
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

        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["retry_context"]["retry_reason"], "low_score")
        self.assertEqual(result["retry_context"]["failed_routes"], ["docs"])
        self.assertEqual(len(result["retry_context"]["preserved_evidence"]), 1)
        self.assertEqual(
            result["retry_context"]["preserved_evidence"][0]["tool"],
            "upload_search",
        )

    def test_validate_evidence_unsupported_claims_falls_back_to_grounded_payload(self) -> None:
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

        self.assertFalse(result["needs_retry"])
        self.assertEqual(result["retry_context"]["retry_reason"], "unsupported_claims")
        self.assertEqual(
            result["response_payload"]["claims"][0]["evidence_ids"],
            ["path:data/notebooks/example.ipynb#cell=0;chunk=0;start=0;end=12"],
        )

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

    def test_summarize_node_records_llm_call_metadata(self) -> None:
        summarize_llm = _CaptureSummaryLLM()
        summarize_node = make_summarize_node(summarize_llm, verbose=False, max_turns=2)

        updates = summarize_node(
            {
                "messages": [
                    HumanMessage(content="u1"),
                    AIMessage(content="a1"),
                    HumanMessage(content="u2"),
                    AIMessage(content="a2"),
                    HumanMessage(content="u3"),
                    AIMessage(content="a3"),
                    HumanMessage(content="u4"),
                ]
            }
        )

        self.assertIn("memory_summary", updates)
        self.assertEqual(len(updates["llm_calls"]), 1)
        self.assertEqual(updates["llm_calls"][0]["stage"], "summarize")
        self.assertEqual(updates["llm_calls"][0]["path"], "direct")

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
        self.assertEqual(updates["response_payload"]["claims"], [])
        self.assertNotIn("llm_calls", updates)

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
        self.assertNotIn("llm_calls", updates)

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
                "messages": [HumanMessage(content="Find groupby in uploaded file")],
                "user_input": "Find groupby in uploaded file",
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
            },
            include_raw=True,
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
        self.assertEqual(len(updates["llm_calls"]), 1)
        self.assertEqual(updates["llm_calls"][0]["path"], "structured")

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

    def test_synthesize_trims_only_conversation_history_and_keeps_fixed_messages(self) -> None:
        capture_llm = _CaptureStructuredSynthesizeLLM(include_raw=True)
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=2)

        updates = synthesize_node(
            {
                "messages": [
                    HumanMessage(content="u1"),
                    AIMessage(content="a1"),
                    HumanMessage(content="u2"),
                    AIMessage(content="a2"),
                    HumanMessage(content="u3"),
                    AIMessage(content="a3"),
                    HumanMessage(content="u4"),
                ],
                "memory_summary": "older summary",
                "user_input": "Explain numpy broadcasting with official docs and save the response.",
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

        self.assertEqual(updates["final_answer"], "synth result")
        sent_messages = capture_llm.last_messages or []
        self.assertEqual(str(sent_messages[0].content), SYS_POLICY)
        self.assertIn("[Conversation Summary]", str(sent_messages[1].content))
        self.assertFalse(
            any(isinstance(message, HumanMessage) and message.content == "u1" for message in sent_messages)
        )
        self.assertTrue(
            any(isinstance(message, HumanMessage) and message.content == "u4" for message in sent_messages)
        )

    def test_synthesize_uses_plain_summary_attach_fallback(self) -> None:
        primary_llm = _StructuredThenPlainFallbackSynthesizeLLM()
        plain_fallback_llm = _CaptureSynthesizeLLM(
            content="NumPy broadcasting expands compatible shapes.\nIt avoids Python-level loops."
        )
        synthesize_node = make_synthesize_node(
            primary_llm,
            llm_synthesizer_compact=plain_fallback_llm,
            verbose=False,
            max_turns=6,
        )

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                "user_input": "Explain numpy broadcasting.",
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
                    },
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://numpy.org/doc/stable/broadcasting-2",
                        "document_id": "url:https://numpy.org/doc/stable/broadcasting-2",
                        "url_or_path": "https://numpy.org/doc/stable/broadcasting-2",
                        "title": "NumPy Docs 2",
                        "snippet": "Broadcasting keeps loops in C.",
                        "score": 0.88,
                    },
                ],
                "synthesis_attempt": 0,
            }
        )

        self.assertEqual(
            updates["response_payload"]["claims"][0]["evidence_ids"],
            ["url:https://numpy.org/doc/stable/"],
        )
        self.assertEqual(
            updates["response_payload"]["claims"][1]["evidence_ids"],
            ["url:https://numpy.org/doc/stable/broadcasting-2"],
        )
        self.assertEqual(updates["llm_calls"][-1]["path"], "plain_summary_attach_fallback")
        synthesis_attempts = [
            item for item in updates["latency_trace"] if item.get("kind") == "synthesis_attempt"
        ]
        self.assertEqual(synthesis_attempts[0]["mode"], "plain_summary_attach_fallback")

    def test_synthesize_falls_back_to_deterministic_grounded_render_when_plain_attach_fails(self) -> None:
        primary_llm = _StructuredThenPlainFallbackSynthesizeLLM()
        failing_plain_llm = _CaptureSynthesizeLLM(content="")
        synthesize_node = make_synthesize_node(
            primary_llm,
            llm_synthesizer_compact=failing_plain_llm,
            verbose=False,
            max_turns=8,
        )

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                "user_input": "Explain numpy broadcasting.",
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

        self.assertIn("Broadcasting expands compatible array shapes.", updates["final_answer"])
        synthesis_attempts = [
            item for item in updates["latency_trace"] if item.get("kind") == "synthesis_attempt"
        ]
        self.assertEqual(synthesis_attempts[0]["mode"], "deterministic_grounded_fallback")

    def test_synthesize_uses_deterministic_grounded_direct_for_upload_routes(self) -> None:
        capture_llm = _CaptureStructuredSynthesizeLLM(include_raw=True)
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=8)

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="Find groupby in uploaded file.")],
                "user_input": "Find groupby in uploaded file.",
                "planner_output": PlannerOutput(
                    use_retrieval=True,
                    tasks=[RetrievalTask(route="upload", query="groupby", k=3)],
                ),
                "retrieved_evidence": [
                    {
                        "kind": "local",
                        "tool": "upload_search",
                        "source_id": "path:uploads/demo/sample.py#chunk=0;start=0;end=48",
                        "document_id": "path:uploads/demo/sample.py",
                        "url_or_path": "uploads/demo/sample.py",
                        "snippet": 'grouped = all_sales.groupby("region")["amount"].sum()',
                        "score": 0.0,
                        "chunk_id": 0,
                        "start_offset": 0,
                        "end_offset": 48,
                    }
                ],
                "synthesis_attempt": 0,
            }
        )

        self.assertIsNone(capture_llm.last_messages)
        self.assertIn("groupby", updates["final_answer"])
        self.assertNotIn("llm_calls", updates)
        synthesis_attempts = [
            item for item in updates["latency_trace"] if item.get("kind") == "synthesis_attempt"
        ]
        self.assertEqual(synthesis_attempts[0]["mode"], "deterministic_grounded_direct")


if __name__ == "__main__":
    unittest.main()
