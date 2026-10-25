"""Malformed observations remain visible without discarding valid evidence."""
from __future__ import annotations

import pytest

from src.app.web.agent_request_support import normalize_debug_info
from src.core.contracts.graph_state import DebugState
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.runtime.agent_runtime.debug_collector import DebugCollector


@pytest.fixture
def debug_payload():
    return DebugCollector().build(
        response={"route_decisions": []}, updated_messages=[], graph_total_ms=200,
    )


def test_invalid_observed_hit_preserves_valid_source_and_reports_incomplete_observation(debug_payload):
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

    normalized = normalize_debug_info(debug_payload, 220, answer_expected=False)

    assert normalized.observed_hits == [hit]
    assert normalized.observability_status == "failed"
    assert "observed_hits" in normalized.missing_required_debug_fields
    assert "DEBUG_NORMALIZATION_FAILED" in normalized.error_codes
    assert any("observed_hits[1]" in error for error in normalized.errors)


@pytest.mark.parametrize("observed_hits", [None, {}, [None]])
def test_invalid_observed_hit_container_is_not_reported_as_an_empty_search(debug_payload, observed_hits):
    debug_payload["observed_hits"] = observed_hits

    normalized = normalize_debug_info(debug_payload, 220, answer_expected=False)

    assert normalized.observed_hits == []
    assert normalized.observability_status == "failed"
    assert "observed_hits" in normalized.missing_required_debug_fields
    assert "DEBUG_NORMALIZATION_FAILED" in normalized.error_codes


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
        updated_messages=[], graph_total_ms=200,
    )
    normalized = normalize_debug_info(debug, 220, answer_expected=False)

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
def test_invalid_serialized_latency_is_reported_as_missing_observation(debug_payload, invalid_latency):
    debug_payload["latency_breakdown"] = invalid_latency

    normalized = normalize_debug_info(debug_payload, 220, answer_expected=False)

    assert normalized.latency_breakdown is None
    assert normalized.observability_status == "failed"
    assert "latency_breakdown" in normalized.missing_required_debug_fields
    assert "DEBUG_NORMALIZATION_FAILED" in normalized.error_codes
    assert any("latency_breakdown" in error for error in normalized.errors)
