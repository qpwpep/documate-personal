import unittest

from pydantic import ValidationError

from src.web.schemas import AgentResponse


class AgentResponseSchemaTest(unittest.TestCase):
    def test_structured_response_payload_is_valid(self) -> None:
        payload = {
            "response": {
                "answer": "hello [1]",
                "claims": [
                    {
                        "text": "hello",
                        "evidence_ids": ["url:https://numpy.org/doc/stable/"],
                        "confidence": 0.8,
                    }
                ],
                "evidence": [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://numpy.org/doc/stable/",
                        "document_id": "url:https://numpy.org/doc/stable/",
                        "url_or_path": "https://numpy.org/doc/stable/",
                        "title": "NumPy Docs",
                        "snippet": "broadcasting rule",
                        "score": 0.99,
                    }
                ],
                "confidence": 0.8,
            },
            "trace": "trace-id",
            "file_path": None,
            "debug": None,
        }
        result = AgentResponse.model_validate(payload)
        self.assertEqual(result.response.answer, "hello [1]")
        self.assertEqual(len(result.response.claims), 1)
        self.assertEqual(result.response.claims[0].evidence_ids, ["url:https://numpy.org/doc/stable/"])
        self.assertEqual(len(result.response.evidence), 1)

    def test_plain_string_response_is_rejected(self) -> None:
        legacy_payload = {
            "response": "legacy string response",
            "trace": "trace-id",
            "file_path": None,
            "debug": None,
        }
        with self.assertRaises(ValidationError):
            AgentResponse.model_validate(legacy_payload)

    def test_debug_retry_context_is_optional_and_parseable(self) -> None:
        payload = {
            "response": {"answer": "uncertain", "claims": [], "evidence": [], "confidence": None},
            "trace": "trace-id",
            "file_path": None,
            "debug": {
                "tool_calls": ["tavily_search"],
                "tool_call_count": 1,
                "errors": ["validate_evidence: retry_reason=unsupported_claims"],
                "observed_evidence": [],
                "retry_context": {
                    "attempt": 1,
                    "max_retries": 1,
                    "retry_reason": "unsupported_claims",
                    "retrieval_feedback": "generated claims referenced unsupported evidence ids",
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                    "retrieval_diagnostic_start_index": 0,
                    "score_avg": None,
                },
            },
        }
        result = AgentResponse.model_validate(payload)
        self.assertIsNotNone(result.debug)
        self.assertIsNotNone(result.debug.retry_context)
        self.assertEqual(result.debug.retry_context.retry_reason, "unsupported_claims")
        self.assertEqual(result.debug.retry_context.retrieval_diagnostic_start_index, 0)

    def test_debug_diagnostics_are_optional_and_parseable(self) -> None:
        payload = {
            "response": {"answer": "follow up", "claims": [], "evidence": [], "confidence": None},
            "trace": "trace-id",
            "file_path": None,
            "debug": {
                "tool_calls": ["tavily_search"],
                "tool_call_count": 1,
                "errors": [],
                "observed_evidence": [],
                "retrieval_diagnostics": [
                    {
                        "tool": "tavily_search",
                        "route": "docs",
                        "status": "error",
                        "message": "invoke failed",
                        "query": "numpy docs",
                        "attempt": 1,
                    }
                ],
                "planner_diagnostics": {
                    "status": "heuristic_fallback",
                    "reason": "planner_failed_or_invalid",
                    "fallback_routes": ["docs"],
                    "intent_required": True,
                    "required_routes": ["docs", "upload"],
                    "override_applied": True,
                    "override_reason": "missing_required_routes",
                },
            },
        }
        result = AgentResponse.model_validate(payload)
        self.assertIsNotNone(result.debug)
        self.assertEqual(result.debug.retrieval_diagnostics[0].status, "error")
        self.assertEqual(result.debug.planner_diagnostics.status, "heuristic_fallback")
        self.assertTrue(result.debug.planner_diagnostics.intent_required)
        self.assertEqual(result.debug.planner_diagnostics.required_routes, ["docs", "upload"])
        self.assertTrue(result.debug.planner_diagnostics.override_applied)
        self.assertEqual(
            result.debug.planner_diagnostics.override_reason,
            "missing_required_routes",
        )

    def test_debug_latency_breakdown_is_optional_and_parseable(self) -> None:
        payload = {
            "response": {"answer": "follow up", "claims": [], "evidence": [], "confidence": None},
            "trace": "trace-id",
            "file_path": None,
            "debug": {
                "tool_calls": ["tavily_search"],
                "tool_call_count": 1,
                "errors": [],
                "observed_evidence": [],
                "latency_ms_server": 1250,
                "latency_breakdown": {
                    "server_total_ms": 1250,
                    "graph_total_ms": 1190,
                    "upload_retriever_build_ms": None,
                    "stage_totals_ms": {
                        "summarize_ms": 0,
                        "planner_ms": 22,
                        "retrieval_total_ms": 810,
                        "synthesis_total_ms": 300,
                        "validation_ms": 40,
                        "action_postprocess_ms": 18,
                    },
                    "stage_attempts": [
                        {"stage": "planner", "attempt": 1, "latency_ms": 22, "status": "llm"},
                        {"stage": "retrieval", "attempt": 1, "latency_ms": 810, "status": "success"},
                    ],
                    "retrieval_routes": [
                        {
                            "route": "docs",
                            "tool": "tavily_search",
                            "attempt": 1,
                            "latency_ms": 790,
                            "status": "success",
                        }
                    ],
                    "synthesis_attempts": [
                        {
                            "attempt": 1,
                            "mode": "structured_only",
                            "structured_ms": 300,
                            "fallback_ms": None,
                            "total_ms": 300,
                        }
                    ],
                },
            },
        }

        result = AgentResponse.model_validate(payload)
        self.assertIsNotNone(result.debug)
        self.assertIsNotNone(result.debug.latency_breakdown)
        self.assertEqual(result.debug.latency_breakdown.graph_total_ms, 1190)
        self.assertEqual(result.debug.latency_breakdown.stage_totals_ms.retrieval_total_ms, 810)
        self.assertEqual(result.debug.latency_breakdown.retrieval_routes[0].route, "docs")
        self.assertEqual(result.debug.latency_breakdown.synthesis_attempts[0].mode, "structured_only")


if __name__ == "__main__":
    unittest.main()
