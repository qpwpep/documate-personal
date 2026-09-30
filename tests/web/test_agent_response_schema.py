import unittest

import pytest
from pydantic import ValidationError

from src.app.web.schemas import AgentResponse
from src.core.contracts.debug import DEBUG_SCHEMA_VERSION
from src.core.uploads import UploadManifest
from tests.web.answer_fixtures import cited_response, response_payload


@pytest.mark.parametrize("invalid_manifest", [
    pytest.param(None, id="null"),
    pytest.param({"epoch": "epoch", "revision": 0}, id="missing-files"),
    pytest.param({"revision": 0, "files": []}, id="missing-epoch"),
    pytest.param({"epoch": "epoch", "files": []}, id="missing-revision"),
    pytest.param({"epoch": "epoch", "revision": "0", "files": []}, id="string-revision"),
    pytest.param({"epoch": "epoch", "revision": True, "files": []}, id="boolean-revision"),
    pytest.param({"epoch": "epoch", "revision": 0.0, "files": []}, id="float-revision"),
    pytest.param({"epoch": "epoch", "revision": 0, "files": None}, id="null-files"),
    pytest.param({"epoch": "epoch", "revision": 0, "files": {}}, id="object-files"),
])
def test_final_response_rejects_incomplete_or_malformed_manifest(invalid_manifest):
    with pytest.raises(ValidationError):
        AgentResponse.model_validate({
            "response": response_payload("answer"), "trace": "trace-id",
            "upload_manifest": invalid_manifest,
        })


def test_final_response_requires_manifest():
    with pytest.raises(ValidationError):
        AgentResponse.model_validate({"response": response_payload("answer"), "trace": "trace-id"})


@pytest.mark.parametrize("invalid_size", ["0", True, 0.0])
def test_final_response_rejects_coerced_nested_file_sizes(invalid_size):
    with pytest.raises(ValidationError):
        AgentResponse.model_validate({
            "response": response_payload("answer"), "trace": "trace-id",
            "upload_manifest": {"epoch": "epoch", "revision": 1, "files": [{
                "file_id": "file", "name": "file.py", "size_bytes": invalid_size,
                "content_hash": "sha256:" + "0" * 64, "source_uri": "upload://file",
            }]},
        })


def test_empty_manifest_serialization_preserves_the_actual_revision():
    manifest = {"epoch": "current-epoch", "revision": 7, "files": []}
    result = AgentResponse.model_validate({
        "response": response_payload("answer"), "trace": "trace-id", "upload_manifest": manifest,
    })
    assert result.upload_manifest == UploadManifest.model_validate(manifest)
    assert result.model_dump(mode="json")["upload_manifest"] == manifest


class AgentResponseSchemaTest(unittest.TestCase):
    def test_structured_response_payload_is_valid(self) -> None:
        expected = cited_response()
        result = AgentResponse.model_validate({
            "response": expected.model_dump(mode="json"), "trace": "trace-id", "debug": None,
            "upload_manifest": {"epoch": "session-epoch", "revision": 0, "files": []},
        })
        self.assertEqual(result.response, expected)
        self.assertNotIn("file_path", result.model_dump())

    def test_plain_string_response_is_rejected(self) -> None:
        legacy_payload = {
            "response": "legacy string response",
            "trace": "trace-id",
            "upload_manifest": {"epoch": "session-epoch", "revision": 0, "files": []},
            "debug": None,
        }
        with self.assertRaises(ValidationError):
            AgentResponse.model_validate(legacy_payload)

    def test_debug_contract_requires_observability_fields(self) -> None:
        payload = {
            "response": response_payload("hello"),
            "trace": "trace-id",
            "upload_manifest": {"epoch": "session-epoch", "revision": 0, "files": []},
            "debug": {
                "tool_calls": [],
                "tool_call_count": 0,
                "errors": [],
                "observed_hits": [],
            },
        }
        with self.assertRaises(ValidationError):
            AgentResponse.model_validate(payload)

    def test_debug_retry_context_is_optional_and_parseable(self) -> None:
        payload = {
            "response": response_payload("uncertain"),
            "trace": "trace-id",
            "upload_manifest": {"epoch": "session-epoch", "revision": 0, "files": []},
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "route_decisions": [],
                "memory_compactions": [],
                "observability_status": "ok",
                "missing_required_debug_fields": [],
                "tool_calls": ["tavily_search"],
                "tool_call_count": 1,
                "errors": ["validate_evidence: retry_reason=unresolved_references"],
                "observed_hits": [],
                "retry_context": {
                    "attempt": 1,
                    "max_retries": 1,
                    "retry_reason": "unresolved_references",
                    "retrieval_feedback": "displayed content referenced unresolved evidence ids",
                    "hit_start_index": 0,
                    "retrieval_error_start_index": 0,
                    "retrieval_diagnostic_start_index": 0,
                    "score_avg": None,
                },
            },
        }
        result = AgentResponse.model_validate(payload)
        self.assertIsNotNone(result.debug)
        self.assertIsNotNone(result.debug.retry_context)
        self.assertEqual(result.debug.retry_context.retry_reason, "unresolved_references")
        self.assertEqual(result.debug.retry_context.retrieval_diagnostic_start_index, 0)

    def test_debug_validation_events_and_route_decisions_are_parseable(self) -> None:
        payload = {
            "response": response_payload("ok"),
            "trace": "trace-id",
            "upload_manifest": {"epoch": "session-epoch", "revision": 0, "files": []},
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "memory_compactions": [],
                "observability_status": "ok",
                "missing_required_debug_fields": [],
                "tool_calls": [],
                "tool_call_count": 0,
                "errors": [],
                "validation_events": ["validate_evidence: retry_reason=unresolved_references"],
                "route_decisions": [
                    {
                        "source": "planner",
                        "sequence": 1,
                        "target": "retrieve_dispatch",
                        "reason": "retrieval_required:2_task(s)",
                    }
                ],
                "observed_hits": [],
            },
        }

        result = AgentResponse.model_validate(payload)

        self.assertIsNotNone(result.debug)
        self.assertEqual(
            result.debug.validation_events,
            ["validate_evidence: retry_reason=unresolved_references"],
        )
        self.assertEqual(result.debug.errors, [])
        self.assertEqual(result.debug.route_decisions[0].target, "retrieve_dispatch")

    def test_debug_diagnostics_are_optional_and_parseable(self) -> None:
        payload = {
            "response": response_payload("follow up"),
            "trace": "trace-id",
            "upload_manifest": {"epoch": "session-epoch", "revision": 0, "files": []},
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "route_decisions": [],
                "memory_compactions": [],
                "observability_status": "ok",
                "missing_required_debug_fields": [],
                "tool_calls": ["tavily_search"],
                "tool_call_count": 1,
                "errors": [],
                "planner_errors": ["planner: structured output invocation failed (boom)"],
                "observed_hits": [],
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
            result.debug.planner_errors,
            ["planner: structured output invocation failed (boom)"],
        )
        self.assertEqual(
            result.debug.planner_diagnostics.override_reason,
            "missing_required_routes",
        )

    def test_debug_latency_breakdown_is_optional_and_parseable(self) -> None:
        payload = {
            "response": response_payload("follow up"),
            "trace": "trace-id",
            "upload_manifest": {"epoch": "session-epoch", "revision": 0, "files": []},
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "route_decisions": [],
                "memory_compactions": [],
                "observability_status": "ok",
                "missing_required_debug_fields": [],
                "tool_calls": ["tavily_search"],
                "tool_call_count": 1,
                "errors": [],
                "observed_hits": [],
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

    def test_debug_latency_breakdown_accepts_deterministic_grounded_direct_mode(self) -> None:
        payload = {
            "response": response_payload("follow up"),
            "trace": "trace-id",
            "upload_manifest": {"epoch": "session-epoch", "revision": 0, "files": []},
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "route_decisions": [],
                "memory_compactions": [],
                "observability_status": "ok",
                "missing_required_debug_fields": [],
                "tool_calls": ["upload_search"],
                "tool_call_count": 1,
                "errors": [],
                "observed_hits": [],
                "latency_ms_server": 120,
                "latency_breakdown": {
                    "server_total_ms": 120,
                    "graph_total_ms": 90,
                    "upload_retriever_build_ms": None,
                    "stage_totals_ms": {
                        "summarize_ms": 0,
                        "planner_ms": 5,
                        "retrieval_total_ms": 40,
                        "synthesis_total_ms": 12,
                        "validation_ms": 20,
                        "action_postprocess_ms": 13,
                    },
                    "stage_attempts": [
                        {"stage": "synthesis", "attempt": 1, "latency_ms": 12, "status": "deterministic_grounded_direct"},
                    ],
                    "retrieval_routes": [],
                    "synthesis_attempts": [
                        {
                            "attempt": 1,
                            "mode": "deterministic_grounded_direct",
                            "structured_ms": 0,
                            "fallback_ms": None,
                            "total_ms": 12,
                        }
                    ],
                },
            },
        }

        result = AgentResponse.model_validate(payload)
        self.assertEqual(result.debug.latency_breakdown.synthesis_attempts[0].mode, "deterministic_grounded_direct")

    def test_debug_canonical_calls_preserve_model_identity_and_usage(self) -> None:
        payload = {
            "response": response_payload("follow up"),
            "trace": "trace-id",
            "upload_manifest": {"epoch": "session-epoch", "revision": 0, "files": []},
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "route_decisions": [],
                "memory_compactions": [],
                "observability_status": "ok",
                "missing_required_debug_fields": [],
                "tool_calls": ["tavily_search"],
                "tool_call_count": 1,
                "errors": [],
                "observed_hits": [],
                "llm_calls": [
                    {
                        "stage": "planner",
                        "attempt": 1,
                        "path": "structured",
                        "model_name": "gpt-5-nano",
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                        "response_metadata": {"model_name": "gpt-5-nano"},
                        "usage_metadata": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                    },
                    {
                        "stage": "synthesis",
                        "attempt": 1,
                        "path": "structured",
                        "model_name": "gpt-5-mini",
                        "usage": {"input_tokens": 20, "output_tokens": 5},
                        "response_metadata": {"model_name": "gpt-5-mini"},
                        "usage_metadata": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
                    },
                ],
            },
        }

        result = AgentResponse.model_validate(payload)
        self.assertIsNotNone(result.debug)
        self.assertEqual([call.model_name for call in result.debug.llm_calls], ["gpt-5-nano", "gpt-5-mini"])
        self.assertEqual(len(result.debug.llm_calls), 2)
        self.assertEqual(result.debug.llm_calls[0].stage, "planner")
        self.assertEqual(result.debug.llm_calls[1].usage.total_tokens, 25)
        self.assertNotIn("token_usage", result.debug.model_dump())


if __name__ == "__main__":
    unittest.main()
