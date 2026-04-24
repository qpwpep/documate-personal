import unittest

from pydantic import ValidationError

from src.core.answer_schema import AgentResponsePayloadModel
from src.core.contracts import PlannerState
from src.core.contracts.boundary.debug import parse_debug_state, parse_retry_state
from src.core.contracts.boundary.graph import normalize_graph_update
from src.core.contracts.boundary.planner import parse_planner_output
from src.core.contracts.boundary.response import parse_response_state
from src.core.contracts.boundary.runtime import parse_session_metadata, parse_slack_destination


class BoundaryAdaptersTest(unittest.TestCase):
    def test_parse_slack_destination_trims_and_discards_empty_values(self) -> None:
        destination = parse_slack_destination(
            {
                "channel_id": "  C123  ",
                "user_id": " ",
                "email": " user@example.com ",
            }
        )

        self.assertEqual(destination.channel_id, "C123")
        self.assertIsNone(destination.user_id)
        self.assertEqual(destination.email, "user@example.com")

    def test_parse_session_metadata_keeps_destination_only_when_present(self) -> None:
        metadata = parse_session_metadata({"slack_destination": {"channel_id": "C123"}})
        empty_metadata = parse_session_metadata({"slack_destination": {"channel_id": " "}})

        self.assertIsNotNone(metadata.slack_destination)
        assert metadata.slack_destination is not None
        self.assertEqual(metadata.slack_destination.channel_id, "C123")
        self.assertIsNone(empty_metadata.slack_destination)

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

    def test_parse_response_state_falls_back_to_empty_payload_for_invalid_raw_payload(self) -> None:
        response = parse_response_state(
            {
                "final_answer": "answer",
                "payload": {"claims": "invalid"},
                "synthesis_attempt": "2",
            }
        )

        self.assertEqual(response.final_answer, "answer")
        self.assertIsInstance(response.payload, AgentResponsePayloadModel)
        self.assertEqual(response.payload.answer, "")
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
                    "session_metadata": {"slack_destination": {"channel_id": "C123"}},
                },
                "retry": {"attempt": 2, "failed_routes": ["upload", "upload"]},
                "messages": "not-a-list",
            }
        )

        self.assertEqual(normalized["runtime"].user_input, "hello")
        assert normalized["runtime"].session_metadata.slack_destination is not None
        self.assertEqual(
            normalized["runtime"].session_metadata.slack_destination.channel_id,
            "C123",
        )
        self.assertEqual(normalized["retry"].attempt, 2)
        self.assertEqual(normalized["retry"].failed_routes, ["upload"])
        self.assertEqual(normalized["messages"], [])

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
        retry_state = parse_retry_state({"retry_scope": "reuse_evidence_resynthesize"})

        self.assertEqual(retry_state.retry_scope, "reuse_evidence_resynthesize")

    def test_parse_retry_state_ignores_invalid_retry_scope(self) -> None:
        retry_state = parse_retry_state({"retry_scope": "unknown"})

        self.assertEqual(retry_state.retry_scope, "refresh_routes")
