import unittest

from pydantic import ValidationError

from src.core.answer_schema import AnswerResponse, finalize_answer, text_document
from src.core.contracts import PlannerState
from src.core.contracts.debug import DebugPayload, PlannerDiagnostic, RetrievalDiagnostic, RetryState
from src.core.contracts.graph_state import DebugState
from src.core.contracts.boundary.debug import parse_debug_payload, parse_debug_state, parse_retry_state
from src.core.contracts.boundary.graph import build_graph_state_input, normalize_graph_update
from src.core.contracts.boundary.planner import parse_planner_output
from src.core.contracts.boundary.response import parse_response_state
from src.core.contracts.boundary.runtime import parse_session_metadata


class BoundaryAdaptersTest(unittest.TestCase):
    def test_graph_input_initializes_route_history_but_partial_updates_do_not(self) -> None:
        self.assertEqual(build_graph_state_input(user_input="hello")["route_decisions"], [])
        self.assertNotIn("route_decisions", normalize_graph_update({"retry": {"attempt": 1}}))

    def test_route_decision_contract_preserves_one_immutable_decision(self) -> None:
        event = {
            "sequence": 3,
            "source": "post_synthesis_validation",
            "target": "synthesize",
            "reason": "unresolved_references",
        }
        normalized = normalize_graph_update({"route_decisions": [event]})
        decision = normalized["route_decisions"][0]
        self.assertEqual(decision.model_dump(), event)
        with self.assertRaises(ValidationError):
            decision.target = "planner"
        output = parse_debug_payload({"route_decisions": [event]})
        self.assertEqual(output.route_decisions, [decision])
        self.assertNotIn("route_decisions", DebugState.model_fields)
        self.assertNotIn("edge_decisions", DebugState.model_fields)
        self.assertNotIn("edge_decisions", DebugPayload.model_fields)

    def test_invalid_route_history_is_rejected_instead_of_silently_dropped(self) -> None:
        valid_event = {
            "sequence": 1,
            "source": "planner",
            "target": "synthesize",
            "reason": "retrieval_not_required",
        }
        for value in (
            None, {}, "not-a-list", [None],
            [{**valid_event, "sequence": 0}],
            [{**valid_event, "sequence": True}],
            [{**valid_event, "sequence": "1"}],
            [{**valid_event, "target": "retry"}],
            [{**valid_event, "source": "unknown"}],
            [{**valid_event, "reason": " "}],
            [{**valid_event, "decision": "retry"}],
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_graph_update({"route_decisions": value})
                with self.assertRaises(ValueError):
                    parse_debug_payload({"route_decisions": value})
        with self.assertRaises(ValueError):
            parse_debug_payload({})

    def test_memory_compaction_diagnostics_survive_internal_and_output_parsing(self) -> None:
        event = {
            "reason": "turn_count",
            "before": {"messages": 8},
            "after": {"messages": 4},
            "removed_messages": 4,
            "summary_fallback": True,
            "fallback_reason": "exception",
        }
        debug = parse_debug_state({"memory_compactions": [event]})
        self.assertEqual(debug.memory_compactions, [event])
        output = parse_debug_payload({"memory_compactions": [event], "route_decisions": []})
        self.assertEqual(output.memory_compactions, [event])

    def test_output_history_never_becomes_a_shadow_history_in_internal_debug(self) -> None:
        event = {
            "sequence": 1,
            "source": "planner",
            "target": "synthesize",
            "reason": "retrieval_not_required",
        }
        output = parse_debug_payload({
            "route_decisions": [event],
            "memory_compactions": [{"removed_messages": 4}],
            "validation_events": ["content_accepted"],
        })
        for external_value in (output, output.model_dump()):
            with self.subTest(value_type=type(external_value).__name__):
                internal = parse_debug_state(external_value)
                self.assertNotIn("route_decisions", internal.model_dump())
                self.assertEqual(internal.validation_events, ["content_accepted"])
                self.assertEqual(internal.memory_compactions, [{"removed_messages": 4}])
                with self.assertRaises(ValueError):
                    parse_debug_payload(internal.model_dump())
                exported = parse_debug_payload({
                    **internal.model_dump(),
                    "route_decisions": output.route_decisions,
                })
                self.assertEqual(exported, output)

    def test_partial_debug_updates_do_not_replace_authoritative_route_history(self) -> None:
        event = {
            "sequence": 7,
            "source": "post_synthesis_validation",
            "target": "action_postprocess",
            "reason": "validation_passed",
        }
        normalized = normalize_graph_update({
            "route_decisions": [event],
            "debug": {"route_decisions": [], "memory_compactions": []},
        })
        self.assertEqual([record.model_dump() for record in normalized["route_decisions"]], [event])
        self.assertNotIn("route_decisions", normalized["debug"].model_dump())

    def test_session_metadata_preserves_one_explicit_recipient(self) -> None:
        metadata = parse_session_metadata({"slack_recipient": {"kind": "channel", "value": " C123 "}})
        self.assertEqual(metadata.slack_recipient.value, "C123")
        self.assertIsNone(parse_session_metadata({}).slack_recipient)

    def test_invalid_explicit_recipient_never_becomes_omitted(self) -> None:
        for value in ({"kind": "user", "value": " "},
                      {"kind": "user", "value": "not-a-user"},
                      {"kind": "channel", "value": "C123", "email": "other@example.com"}):
            with self.assertRaises(ValidationError):
                parse_session_metadata({"slack_recipient": value})
        with self.assertRaises(ValidationError):
            parse_session_metadata({"slack_destination": {"channel_id": "C123"}})

    def test_parse_planner_output_falls_back_and_records_error(self) -> None:
        errors: list[str] = []

        output = parse_planner_output(
            {
                "use_retrieval": False,
                "tasks": [{"route": "docs", "query": "numpy", "k": 4}],
            },
            errors,
        )

        self.assertFalse(output.use_retrieval)
        self.assertEqual(output.tasks, [])
        self.assertEqual(len(errors), 1)

    def test_parse_response_state_rejects_invalid_or_replaced_contracts(self) -> None:
        with self.assertRaises(ValidationError):
            parse_response_state({"payload": {"claims": "invalid"}})
        with self.assertRaises(ValidationError):
            parse_response_state({"result": {"content": {"blocks": "invalid"}}})

    def test_parse_response_state_preserves_checked_document(self) -> None:
        result = finalize_answer(text_document("answer"), [])
        response = parse_response_state({"result": result.model_dump(), "synthesis_attempt": 2})
        self.assertEqual(response.result, result)
        self.assertEqual(response.synthesis_attempt, 2)

    def test_parse_debug_state_normalizes_nested_retry_and_messages(self) -> None:
        debug = parse_debug_state(
            {
                "tool_calls": ["tavily_search"],
                "retry_context": {
                    "needs_retry": True,
                    "attempt": 1,
                    "failed_routes": ["docs", "docs", "unknown"],
                },
                "llm_calls": [
                    {
                        "stage": "planner",
                        "attempt": 1,
                        "path": "structured",
                        "response_metadata": {"model_name": "gpt-5-mini"},
                        "usage_metadata": {"input_tokens": 1},
                    }
                ],
            }
        )

        self.assertEqual(debug.tool_calls, ["tavily_search"])
        assert debug.retry_context is not None
        self.assertEqual(debug.retry_context.failed_routes, ["docs"])
        self.assertEqual(len(debug.llm_calls), 1)

    def test_normalize_graph_update_parses_partial_state(self) -> None:
        normalized = normalize_graph_update(
            {
                "runtime": {
                    "user_input": "hello",
                    "session_metadata": {"slack_recipient": {"kind": "channel", "value": "C123"}},
                },
                "retry": {"attempt": 2, "failed_routes": ["upload", "upload"]},
                "messages": "not-a-list",
            }
        )

        self.assertEqual(normalized["runtime"].user_input, "hello")
        assert normalized["runtime"].session_metadata.slack_recipient is not None
        self.assertEqual(
            normalized["runtime"].session_metadata.slack_recipient.value,
            "C123",
        )
        self.assertEqual(normalized["retry"].attempt, 2)
        self.assertEqual(normalized["retry"].failed_routes, ["upload"])
        self.assertEqual(normalized["messages"], [])

    def test_debug_payload_preserves_local_records_when_reading_legacy_benchmarks(self) -> None:
        debug = parse_debug_state(
            {
                "error_codes": ["RAG_INDEX_MISSING"],
                "planner_diagnostics": {
                    "required_routes": ["local", "docs", "local", "unknown"],
                    "fallback_routes": ["local"],
                },
                "retry_context": {"failed_routes": ["local", "unknown", "local"]},
                "retrieval_diagnostics": [
                    {
                        "tool": "rag_search",
                        "route": "local",
                        "status": "unavailable",
                        "error_code": "RAG_INDEX_MISSING",
                    }
                ],
            }
        )

        self.assertEqual(
            debug,
            DebugState(
                error_codes=["RAG_INDEX_MISSING"],
                planner_diagnostics=PlannerDiagnostic(
                    required_routes=["docs", "local"],
                    fallback_routes=["local"],
                ),
                retry_context=RetryState(failed_routes=["local"]),
                retrieval_diagnostics=[
                    RetrievalDiagnostic(
                        tool="rag_search",
                        route="local",
                        status="unavailable",
                        error_code="RAG_INDEX_MISSING",
                    )
                ],
            ),
        )

    def test_typed_state_construction_raises_instead_of_silent_fallback(self) -> None:
        with self.assertRaises(ValidationError):
            PlannerState(
                output={
                    "use_retrieval": False,
                    "tasks": [{"route": "docs", "query": "numpy", "k": 4}],
                }
            )

    def test_parse_retry_state_preserves_zero_defaults(self) -> None:
        retry_state = parse_retry_state({"score_avg": None, "failed_routes": []})

        self.assertIsNone(retry_state.score_avg)
        self.assertEqual(retry_state.failed_routes, [])

    def test_parse_retry_state_preserves_valid_retry_scope(self) -> None:
        retry_state = parse_retry_state({"retry_scope": "reuse_hits_resynthesize"})

        self.assertEqual(retry_state.retry_scope, "reuse_hits_resynthesize")

    def test_parse_retry_state_ignores_invalid_retry_scope(self) -> None:
        retry_state = parse_retry_state({"retry_scope": "unknown"})

        self.assertEqual(retry_state.retry_scope, "refresh_routes")
