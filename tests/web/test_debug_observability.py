"""Runtime diagnostics are checked once before HTTP serialization."""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import ValidationError

from src.app.web.agent_request_support import build_debug_info
from src.core.answer_schema import ActionReceipt, finalize_answer, text_document
from src.core.contracts.graph_state import DebugState, ResponseState, RetrievalState
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.eval.online_runner.response_parser import parse_agent_response
from src.runtime.agent_runtime.debug_collector import DebugCollector


@pytest.fixture
def debug_payload():
    return DebugCollector().build(
        response={"route_decisions": []}, graph_total_ms=200,
    )


def test_canonical_error_codes_survive_runtime_and_web_boundaries():
    state = DebugState(error_codes=[
        "provider_unavailable", "model_output_invalid", "LOCAL_RAG_FAILED", "provider_unavailable",
    ])
    result = finalize_answer(text_document("Saved answer"), [], actions=[ActionReceipt(
        kind="save_text", status="error", error="Save failed", error_code="save_failed", verification="failed",
    )])
    debug = DebugCollector().build(
        response={"route_decisions": [], "debug": state, "response": ResponseState(result=result)},
        graph_total_ms=200,
    )
    expected = ["provider_unavailable", "model_output_invalid", "LOCAL_RAG_FAILED", "save_failed"]

    assert debug["error_codes"] == expected
    assert debug["action_results"]["save_text"]["error_code"] == "save_failed"
    normalized = build_debug_info(debug, 220, answer_expected=False)
    assert normalized.error_codes == expected
    assert normalized.action_results.save_text.error_code == "save_failed"


def test_web_rejects_noncanonical_internal_codes_while_eval_parses_external_observations(debug_payload):
    debug_payload["error_codes"] = [
        " provider_unavailable ", "model_output_invalid", "save_failed", "LOCAL_RAG_FAILED",
        "provider_unavailable", "unknown", None,
    ]
    expected = ["provider_unavailable", "model_output_invalid", "save_failed", "LOCAL_RAG_FAILED"]

    with pytest.raises(ValidationError):
        build_debug_info(debug_payload, 220, answer_expected=False)
    parsed = parse_agent_response({"status": "needs_input", "message": "Which source?", "debug": debug_payload})
    assert parsed.error_codes == expected
    assert parsed.response_errors == []


def test_current_search_observations_do_not_depend_on_conversation_tool_messages():
    hits = []
    for name, content in (("current", "Current source text"), ("previous", "Previous source text")):
        snapshot = build_snapshot(
            source_uri=f"https://example.test/{name}", title=name, media_type="text/plain",
            source_type="official", content=content, parser="test", parser_version="1",
        )
        hits.append(SearchHit(evidence=build_evidence(
            snapshot=snapshot, element=DocumentElement(element_id="paragraph", kind="paragraph", text=content),
        ), score=RetrievalScore(metric="relevance", raw=0.9, direction="higher"), rank=1))
    debug = DebugCollector().build(
        response={"route_decisions": [], "retrieval": RetrievalState(hit_log=[hits[0]]), "messages": [
            HumanMessage(content="Previous question"),
            ToolMessage(name="tavily_search", tool_call_id="previous-search",
                        content=json.dumps({"hits": [hits[1].model_dump(mode="json")]})),
            HumanMessage(content="Current question"),
        ]},
        graph_total_ms=200,
    )

    normalized = build_debug_info(debug, 220, answer_expected=False)

    assert [hit.evidence.excerpt for hit in normalized.observed_hits] == ["Current source text"]
    assert "observed_hits" not in normalized.missing_required_debug_fields


def test_invalid_internal_hit_is_rejected_instead_of_being_dropped(debug_payload):
    snapshot = build_snapshot(
        source_uri="https://example.test/reference", title="Reference", media_type="text/plain",
        source_type="official", content="Known source text", parser="test", parser_version="1",
    )
    hit = SearchHit(
        evidence=build_evidence(
            snapshot=snapshot, element=DocumentElement(element_id="paragraph", kind="paragraph", text="Known source text"),
        ),
        score=RetrievalScore(metric="relevance", raw=0.9, direction="higher"), rank=1,
    )
    debug_payload["observed_hits"] = [hit.model_dump(mode="json"), {}]

    with pytest.raises(ValidationError):
        build_debug_info(debug_payload, 220, answer_expected=False)


@pytest.mark.parametrize("observed_hits", [None, {}, [None]])
def test_invalid_observed_hit_container_is_not_reported_as_an_empty_search(debug_payload, observed_hits):
    debug_payload["observed_hits"] = observed_hits

    with pytest.raises(ValidationError):
        build_debug_info(debug_payload, 220, answer_expected=False)


@pytest.mark.parametrize("invalid_event", [
    {"kind": "stage", "stage": "planner", "latency_ms": "broken"},
    {"kind": "retrieval_route", "route": "docs", "latency_ms": "broken"},
    {"kind": "synthesis_attempt", "mode": "structured_only", "total_ms": "broken"},
])
def test_invalid_latency_preserves_valid_measurement_and_marks_debug_incomplete(invalid_event):
    debug = DebugCollector().build(
        response={
            "route_decisions": [],
            "debug": DebugState(latency_trace=[
                {"kind": "stage", "stage": "planner", "attempt": 1, "latency_ms": 120},
                invalid_event,
            ]),
        },
        graph_total_ms=200,
    )
    normalized = build_debug_info(debug, 220, answer_expected=False)

    assert normalized.latency_breakdown.stage_totals_ms.planner_ms == 120
    assert len(normalized.latency_breakdown.stage_attempts) == 1
    assert normalized.latency_breakdown.graph_total_ms == 200
    assert normalized.latency_breakdown.server_total_ms == 220
    assert normalized.observability_status == "failed"
    assert "latency_breakdown" in normalized.missing_required_debug_fields
    assert "DEBUG_NORMALIZATION_FAILED" in normalized.error_codes
    assert any("latency_trace[1]" in error for error in normalized.errors)


@pytest.mark.parametrize("invalid_latency", [
    {"stage_totals_ms": {"planner_ms": "broken"}},
    "broken",
])
def test_invalid_internal_latency_is_rejected(debug_payload, invalid_latency):
    debug_payload["latency_breakdown"] = invalid_latency

    with pytest.raises(ValidationError):
        build_debug_info(debug_payload, 220, answer_expected=False)


@pytest.mark.parametrize("field,value", [("observability_status", "unknown"), ("schema_version", "14")])
def test_internal_metadata_is_not_silently_normalized(debug_payload, field, value):
    debug_payload[field] = value

    with pytest.raises(ValidationError):
        build_debug_info(debug_payload, 220, answer_expected=False)


def test_missing_internal_observation_is_not_defaulted_to_empty(debug_payload):
    del debug_payload["observed_hits"]

    with pytest.raises(ValueError, match="observed_hits"):
        build_debug_info(debug_payload, 220, answer_expected=False)


def test_invalid_execution_history_cannot_bypass_the_typed_contract(debug_payload):
    debug_payload["execution_evidence"] = {
        "schema_version": 1, "request_id": "test-request", "status": "complete",
        "events": [{"sequence": 1, "invocation_id": "save-1", "tool_name": "save_text", "phase": "started"}],
    }

    with pytest.raises(ValidationError, match="unfinished"):
        build_debug_info(debug_payload, 220, answer_expected=False)
