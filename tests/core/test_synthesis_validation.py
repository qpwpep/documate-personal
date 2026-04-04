import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.nodes.session import make_summarize_node
from src.nodes.synthesis import make_synthesize_node
from src.nodes.validation import make_validate_evidence_node
from src.planner_schema import PlannerOutput, RetrievalTask
from src.prompts import SYS_POLICY

from .helpers import (
    _CaptureStructuredSynthesizeLLM,
    _CaptureSummaryLLM,
    _CaptureSynthesizeLLM,
    _StructuredThenPlainFallbackSynthesizeLLM,
    build_legacy_state,
)


def _state(payload: dict):
    return build_legacy_state(payload)


def _retry(result):
    return result["retry"]


def _response(result):
    return result["response"]


def _debug(result):
    return result["debug"]


def _docs_evidence(
    *,
    score: float = 0.9,
    source_id: str = "url:https://numpy.org/doc/stable/",
    snippet: str = "Broadcasting expands compatible array shapes.",
):
    return {
        "kind": "official",
        "tool": "tavily_search",
        "source_id": source_id,
        "document_id": source_id,
        "url_or_path": source_id.removeprefix("url:"),
        "title": "NumPy Docs",
        "snippet": snippet,
        "score": score,
    }


def _local_evidence(
    *,
    tool: str = "rag_search",
    source_id: str = "path:data/notebooks/example.ipynb#cell=0;chunk=0;start=0;end=12",
    path: str = "data/notebooks/example.ipynb",
    snippet: str = "example snippet",
    score: float = 0.9,
):
    return {
        "kind": "local",
        "tool": tool,
        "source_id": source_id,
        "document_id": source_id.split("#", 1)[0],
        "url_or_path": path,
        "snippet": snippet,
        "score": score,
        "cell_id": 0,
        "chunk_id": 0,
        "start_offset": 0,
        "end_offset": 12,
    }


class SynthesisValidationTest(unittest.TestCase):
    def test_validate_evidence_retries_once_for_docs_only_no_evidence(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)])

        result = validate_node(
            _state(
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
        )

        self.assertTrue(_retry(result).needs_retry)
        self.assertEqual(_retry(result).attempt, 1)
        self.assertEqual(_retry(result).retry_reason, "no_evidence")
        self.assertNotIn("response", result)

    def test_validate_evidence_sets_tool_error_reason(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)])
        result = validate_node(
            _state(
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
        )

        self.assertTrue(_retry(result).needs_retry)
        self.assertEqual(_retry(result).attempt, 1)
        self.assertEqual(_retry(result).retry_reason, "tool_error")
        self.assertNotIn("response", result)

    def test_validate_evidence_does_not_treat_planner_errors_as_tool_errors(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)])
        result = validate_node(
            _state(
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
        )

        self.assertTrue(_retry(result).needs_retry)
        self.assertEqual(_retry(result).retry_reason, "no_evidence")

    def test_validate_evidence_retries_docs_only_tool_error_even_with_grounded_payload(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)])
        result = validate_node(
            _state(
                {
                    "planner_output": planner_output,
                    "retrieved_evidence": [_docs_evidence()],
                    "retrieval_errors": ["tavily_search: failed (timeout)"],
                    "response_payload": {"answer": "draft", "claims": [], "evidence": [], "confidence": None},
                    "retry_context": {
                        "attempt": 0,
                        "max_retries": 1,
                        "evidence_start_index": 0,
                        "retrieval_error_start_index": 0,
                        "retrieval_diagnostic_start_index": 0,
                    },
                }
            )
        )

        self.assertTrue(_retry(result).needs_retry)
        self.assertEqual(_retry(result).attempt, 1)
        self.assertEqual(_retry(result).retry_reason, "tool_error")
        self.assertNotIn("response", result)

    def test_validate_evidence_maps_upload_unavailable_to_blocked_missing_upload(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="groupby", k=3)])
        result = validate_node(
            _state(
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
        )

        self.assertFalse(_retry(result).needs_retry)
        self.assertEqual(_retry(result).retry_reason, "blocked_missing_upload")
        self.assertIn("response", result)

    def test_validate_evidence_retries_docs_only_low_score(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)])
        result = validate_node(
            _state(
                {
                    "planner_output": planner_output,
                    "retrieved_evidence": [_docs_evidence(score=0.2, snippet="official docs")],
                    "response_payload": {"answer": "draft", "claims": [], "evidence": [], "confidence": None},
                    "retry_context": {
                        "attempt": 0,
                        "max_retries": 1,
                        "evidence_start_index": 0,
                        "retrieval_error_start_index": 0,
                    },
                }
            )
        )

        self.assertTrue(_retry(result).needs_retry)
        self.assertEqual(_retry(result).attempt, 1)
        self.assertEqual(_retry(result).retry_reason, "low_score")
        self.assertAlmostEqual(_retry(result).score_avg, 0.2)
        self.assertNotIn("response", result)

    def test_validate_evidence_salvages_upload_low_score_without_retry(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="groupby", k=3)])
        result = validate_node(
            _state(
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
                    "response_payload": {"answer": "draft", "claims": [], "evidence": [], "confidence": None},
                    "retry_context": {
                        "attempt": 0,
                        "max_retries": 1,
                        "evidence_start_index": 0,
                        "retrieval_error_start_index": 0,
                    },
                }
            )
        )

        self.assertFalse(_retry(result).needs_retry)
        self.assertEqual(_retry(result).retry_reason, "unsupported_claims")
        self.assertIn("groupby", _response(result).final_answer)

    def test_validate_evidence_retries_docs_half_of_docs_upload_and_preserves_upload_context(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[
                RetrievalTask(route="docs", query="train_test_split official docs", k=3),
                RetrievalTask(route="upload", query="uploaded notebook example", k=3),
            ],
        )
        result = validate_node(
            _state(
                {
                    "planner_output": planner_output,
                    "retrieved_evidence": [
                        _docs_evidence(score=0.1, source_id="url:https://huggingface.co/docs/bad", snippet="unrelated content"),
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
                            "query": "train_test_split official docs",
                            "attempt": 1,
                        },
                        {
                            "tool": "upload_search",
                            "route": "upload",
                            "status": "success",
                            "message": "",
                            "query": "uploaded notebook example",
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
        )

        self.assertTrue(_retry(result).needs_retry)
        self.assertEqual(_retry(result).retry_reason, "low_score")
        self.assertEqual(_retry(result).failed_routes, ["docs"])
        self.assertEqual(len(_retry(result).preserved_evidence), 1)
        self.assertEqual(_retry(result).preserved_evidence[0]["tool"], "upload_search")

    def test_validate_evidence_unsupported_claims_falls_back_to_grounded_payload(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="local", query="example", k=3)])
        valid_source = "path:data/notebooks/example.ipynb#cell=0;chunk=0;start=0;end=12"
        result = validate_node(
            _state(
                {
                    "planner_output": planner_output,
                    "retrieved_evidence": [_local_evidence(source_id=valid_source)],
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
        )

        self.assertFalse(_retry(result).needs_retry)
        self.assertEqual(_retry(result).retry_reason, "unsupported_claims")
        self.assertEqual(_response(result).payload.claims[0].evidence_ids, [valid_source])

    def test_validate_evidence_unsupported_claims_rebalances_hybrid_routes(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[
                RetrievalTask(route="docs", query="train_test_split official docs", k=3),
                RetrievalTask(route="upload", query="uploaded notebook example", k=3),
            ],
        )
        result = validate_node(
            _state(
                {
                    "planner_output": planner_output,
                    "retrieved_evidence": [
                        _docs_evidence(
                            source_id="url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
                            snippet="Split arrays or matrices into random train and test subsets.",
                        ),
                        _local_evidence(
                            tool="upload_search",
                            source_id="path:uploads/demo/sample.ipynb#cell=2;chunk=0;start=0;end=64",
                            path="uploads/demo/sample.ipynb",
                            snippet="X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
                            score=0.0,
                        ),
                    ],
                    "response_payload": {
                        "answer": "local-only answer",
                        "claims": [
                            {
                                "text": "업로드 노트북에서는 test_size=0.2와 random_state=42를 사용합니다.",
                                "evidence_ids": ["path:uploads/demo/sample.ipynb#cell=2;chunk=0;start=0;end=64"],
                                "confidence": 0.8,
                            },
                            {
                                "text": "train_test_split 공식 문법은 train_size와 stratify도 지원합니다.",
                                "evidence_ids": ["url:https://missing.example.com/train_test_split"],
                                "confidence": 0.6,
                            },
                        ],
                        "evidence": [],
                        "confidence": 0.7,
                    },
                    "retry_context": {
                        "attempt": 0,
                        "max_retries": 1,
                        "evidence_start_index": 0,
                        "retrieval_error_start_index": 0,
                    },
                }
            )
        )

        self.assertFalse(_retry(result).needs_retry)
        self.assertEqual(_retry(result).retry_reason, "unsupported_claims")
        self.assertIn("공식 문서 기준:", _response(result).final_answer)
        self.assertIn("반면 업로드 파일에서는", _response(result).final_answer)
        self.assertEqual(
            [item.tool for item in _response(result).payload.evidence],
            ["tavily_search", "upload_search"],
        )

    def test_validate_evidence_filters_to_valid_claims_after_retry_budget(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="local", query="example", k=3)])
        valid_source = "path:data/notebooks/example.ipynb#cell=0;chunk=0;start=0;end=12"
        result = validate_node(
            _state(
                {
                    "planner_output": planner_output,
                    "retrieved_evidence": [_local_evidence(source_id=valid_source)],
                    "response_payload": {
                        "answer": "kept [1] dropped [2]",
                        "claims": [
                            {"text": "kept", "evidence_ids": [valid_source], "confidence": 0.9},
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
        )

        self.assertFalse(_retry(result).needs_retry)
        self.assertEqual(_response(result).final_answer, "kept [1]")
        self.assertEqual(len(_response(result).payload.claims), 1)
        self.assertEqual(len(_response(result).payload.evidence), 1)

    def test_validate_evidence_passes_when_retrieval_not_required(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=False, tasks=[])
        result = validate_node(_state({"planner_output": planner_output, "retrieved_evidence": [], "synthesis_attempt": 1}))
        self.assertFalse(_retry(result).needs_retry)

    def test_summarize_node_records_llm_call_metadata(self) -> None:
        summarize_llm = _CaptureSummaryLLM()
        summarize_node = make_summarize_node(summarize_llm, verbose=False, max_turns=2)

        updates = summarize_node(
            _state(
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
        )

        self.assertIn("runtime", updates)
        self.assertEqual(updates["runtime"].memory_summary, "summary line")
        self.assertEqual(len(_debug(updates).llm_calls), 1)
        self.assertEqual(_debug(updates).llm_calls[0].stage, "summarize")
        self.assertEqual(_debug(updates).llm_calls[0].path, "direct")

    def test_synthesize_node_keeps_sys_policy_persona(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                    "retrieved_evidence": [],
                    "synthesis_attempt": 0,
                }
            )
        )
        self.assertEqual(_response(updates).final_answer, "synth result")
        self.assertIsNotNone(capture_llm.last_messages)
        self.assertIsInstance(capture_llm.last_messages[0], SystemMessage)
        self.assertEqual(capture_llm.last_messages[0].content, SYS_POLICY)

    def test_synthesize_action_only_save_request_builds_deterministic_artifact_when_no_previous_answer_exists(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="save this answer to txt")],
                    "user_input": "save this answer to txt",
                    "retrieved_evidence": [],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertIsNone(capture_llm.last_messages)
        self.assertIn("저장 내용", _response(updates).final_answer)
        self.assertIn("이 메시지를 그대로 텍스트 파일에 저장합니다.", _response(updates).final_answer)
        self.assertEqual(_response(updates).payload.claims, [])

    def test_synthesize_action_only_slack_requests_destination_without_metadata(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="send this to slack")],
                    "user_input": "send this to slack",
                    "retrieved_evidence": [],
                    "synthesis_attempt": 0,
                    "session_metadata": {"slack_destination": None},
                }
            )
        )

        self.assertIsNone(capture_llm.last_messages)
        self.assertIn("channel_id", _response(updates).final_answer)

    def test_synthesize_action_only_slack_uses_session_metadata_without_followup(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            _state(
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
        )

        self.assertIsNone(capture_llm.last_messages)
        self.assertEqual(_response(updates).final_answer, "previous answer")
        self.assertEqual(_response(updates).payload.answer, "previous answer")

    def test_synthesize_short_circuits_guided_followup(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)
        followup = "업로드한 파일을 먼저 올려 주세요."

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="Find groupby in uploaded file")],
                    "user_input": "Find groupby in uploaded file",
                    "guided_followup": followup,
                    "retrieved_evidence": [],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertIsNone(capture_llm.last_messages)
        self.assertEqual(_response(updates).final_answer, followup)

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
            _state(
                {
                    "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                    "retrieved_evidence": [_docs_evidence()],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertEqual(_response(updates).final_answer, "Broadcasting expands compatible array shapes. [1]")
        self.assertEqual(len(_response(updates).payload.claims), 1)
        self.assertEqual(len(_response(updates).payload.evidence), 1)
        self.assertEqual(len(_debug(updates).llm_calls), 1)
        self.assertEqual(_debug(updates).llm_calls[0].path, "structured")

    def test_synthesize_empty_structured_output_falls_back_to_non_blank_answer(self) -> None:
        capture_llm = _CaptureStructuredSynthesizeLLM(
            {
                "answer": "",
                "claims": [],
                "confidence": None,
            },
            include_raw=True,
        )
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=8)

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="1")],
                    "user_input": "1",
                    "retrieved_evidence": [],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertTrue(_response(updates).final_answer.strip())
        self.assertEqual(_response(updates).payload.answer, _response(updates).final_answer)
        self.assertIn("structured output was empty", _debug(updates).synthesis_errors[0])

    def test_synthesize_uses_only_current_attempt_evidence_window(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=8)

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                    "retrieved_evidence": [
                        _docs_evidence(source_id="url:https://old.example.com/", snippet="old snippet"),
                        _docs_evidence(source_id="url:https://new.example.com/", snippet="new snippet"),
                    ],
                    "retry_context": {"evidence_start_index": 1},
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertEqual(_response(updates).final_answer, "synth result")
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
            _state(
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
                    "retrieved_evidence": [_docs_evidence()],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertEqual(_response(updates).final_answer, "synth result")
        sent_messages = capture_llm.last_messages or []
        self.assertEqual(str(sent_messages[0].content), SYS_POLICY)
        self.assertIn("[Conversation Summary]", str(sent_messages[1].content))
        self.assertFalse(any(isinstance(message, HumanMessage) and message.content == "u1" for message in sent_messages))
        self.assertTrue(any(isinstance(message, HumanMessage) and message.content == "u4" for message in sent_messages))

    def test_synthesize_truncates_long_evidence_snippets_in_prompt(self) -> None:
        capture_llm = _CaptureStructuredSynthesizeLLM(include_raw=True)
        synthesize_node = make_synthesize_node(
            capture_llm,
            verbose=False,
            max_turns=6,
            prompt_snippet_char_limit=40,
        )
        long_snippet = "Broadcasting expands compatible array shapes across dimensions. " * 4

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                    "retrieved_evidence": [_docs_evidence(snippet=long_snippet)],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertEqual(_response(updates).final_answer, "synth result")
        retrieved_evidence_messages = [
            str(message.content)
            for message in (capture_llm.last_messages or [])
            if isinstance(message, SystemMessage) and "[Retrieved Evidence]" in str(message.content)
        ]
        self.assertEqual(len(retrieved_evidence_messages), 1)
        self.assertIn("Broadcasting expa ... across dimensions.", retrieved_evidence_messages[0])
        self.assertNotIn(long_snippet.strip(), retrieved_evidence_messages[0])

    def test_synthesize_uses_local_deterministic_fallback_after_structured_failure(self) -> None:
        primary_llm = _StructuredThenPlainFallbackSynthesizeLLM()
        synthesize_node = make_synthesize_node(
            primary_llm,
            verbose=False,
            max_turns=6,
        )

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                    "user_input": "Explain numpy broadcasting.",
                    "retrieved_evidence": [
                        _docs_evidence(),
                        _docs_evidence(
                            source_id="url:https://numpy.org/doc/stable/broadcasting-2",
                            snippet="Broadcasting keeps loops in C.",
                        ),
                    ],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertEqual(_response(updates).payload.claims[0].evidence_ids, ["url:https://numpy.org/doc/stable/"])
        self.assertEqual(
            _response(updates).payload.claims[1].evidence_ids,
            ["url:https://numpy.org/doc/stable/broadcasting-2"],
        )
        self.assertEqual(len(_debug(updates).llm_calls), 1)
        self.assertEqual(_debug(updates).llm_calls[-1].path, "structured")
        synthesis_attempts = [
            item for item in _debug(updates).latency_trace if item.get("kind") == "synthesis_attempt"
        ]
        self.assertEqual(synthesis_attempts[0]["mode"], "deterministic_grounded_fallback")

    def test_synthesize_deterministic_fallback_strips_docs_navigation_chrome(self) -> None:
        primary_llm = _StructuredThenPlainFallbackSynthesizeLLM()
        synthesize_node = make_synthesize_node(
            primary_llm,
            verbose=False,
            max_turns=6,
        )

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                    "user_input": "Explain numpy broadcasting.",
                    "retrieved_evidence": [
                        _docs_evidence(
                            snippet=(
                                "# Broadcasting\n"
                                "Home > Docs > API\n"
                                "Table of contents\n"
                                "Broadcasting expands compatible array shapes across dimensions.\n"
                                "Previous: Intro"
                            )
                        )
                    ],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertIn(
            "Broadcasting expands compatible array shapes across dimensions.",
            _response(updates).final_answer,
        )
        self.assertNotIn("Home > Docs > API", _response(updates).final_answer)
        self.assertNotIn("Table of contents", _response(updates).final_answer)
        self.assertNotIn("Previous: Intro", _response(updates).final_answer)
        self.assertNotIn("# Broadcasting", _response(updates).final_answer)

    def test_synthesize_does_not_invoke_secondary_llm_for_fallback(self) -> None:
        primary_llm = _StructuredThenPlainFallbackSynthesizeLLM()
        failing_plain_llm = _CaptureSynthesizeLLM(content="should not be used")
        synthesize_node = make_synthesize_node(
            primary_llm,
            llm_synthesizer_compact=failing_plain_llm,
            verbose=False,
            max_turns=8,
        )

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                    "user_input": "Explain numpy broadcasting.",
                    "planner_output": PlannerOutput(
                        use_retrieval=True,
                        tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
                    ),
                    "retrieved_evidence": [_docs_evidence()],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertIn("Broadcasting expands compatible array shapes.", _response(updates).final_answer)
        self.assertIsNone(failing_plain_llm.last_messages)
        synthesis_attempts = [
            item for item in _debug(updates).latency_trace if item.get("kind") == "synthesis_attempt"
        ]
        self.assertEqual(synthesis_attempts[0]["mode"], "deterministic_grounded_fallback")

    def test_synthesize_uses_deterministic_grounded_direct_for_explicit_upload_extraction(self) -> None:
        capture_llm = _CaptureStructuredSynthesizeLLM(include_raw=True)
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=8)

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="Extract the exact groupby code snippet from the uploaded file.")],
                    "user_input": "Extract the exact groupby code snippet from the uploaded file.",
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
        )

        self.assertIsNone(capture_llm.last_messages)
        self.assertIn("groupby", _response(updates).final_answer)
        self.assertEqual(len(_debug(updates).llm_calls), 0)
        synthesis_attempts = [
            item for item in _debug(updates).latency_trace if item.get("kind") == "synthesis_attempt"
        ]
        self.assertEqual(synthesis_attempts[0]["mode"], "deterministic_grounded_direct")

    def test_synthesize_keeps_multiple_upload_evidence_items_for_non_extraction_requests(self) -> None:
        capture_llm = _CaptureStructuredSynthesizeLLM(
            {
                "answer": "train_test_split uses test_size=0.2 and random_state=42.",
                "claims": [
                    {
                        "text": "train_test_split uses test_size=0.2 and random_state=42.",
                        "evidence_ids": [
                            "path:uploads/demo/sample.ipynb#cell=2;chunk=0;start=0;end=64",
                        ],
                        "confidence": 0.8,
                    }
                ],
                "confidence": 0.8,
            },
            include_raw=True,
        )
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=8)

        updates = synthesize_node(
            _state(
                {
                    "messages": [HumanMessage(content="Find the train_test_split parameters in the uploaded notebook.")],
                    "user_input": "Find the train_test_split parameters in the uploaded notebook.",
                    "planner_output": PlannerOutput(
                        use_retrieval=True,
                        tasks=[RetrievalTask(route="upload", query="train_test_split", k=3)],
                    ),
                    "retrieved_evidence": [
                        {
                            "kind": "local",
                            "tool": "upload_search",
                            "source_id": "path:uploads/demo/sample.ipynb#cell=1;chunk=0;start=0;end=48",
                            "document_id": "path:uploads/demo/sample.ipynb",
                            "url_or_path": "uploads/demo/sample.ipynb",
                            "snippet": "from sklearn.model_selection import train_test_split",
                            "score": 0.2,
                            "cell_id": 1,
                            "chunk_id": 0,
                            "start_offset": 0,
                            "end_offset": 48,
                        },
                        {
                            "kind": "local",
                            "tool": "upload_search",
                            "source_id": "path:uploads/demo/sample.ipynb#cell=2;chunk=0;start=0;end=64",
                            "document_id": "path:uploads/demo/sample.ipynb",
                            "url_or_path": "uploads/demo/sample.ipynb",
                            "snippet": "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
                            "score": 0.1,
                            "cell_id": 2,
                            "chunk_id": 0,
                            "start_offset": 0,
                            "end_offset": 64,
                        },
                    ],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertIsNotNone(capture_llm.last_messages)
        self.assertEqual(
            _response(updates).final_answer,
            "train_test_split uses test_size=0.2 and random_state=42. [1]",
        )
