"""Committed routing history survives each observation boundary unchanged."""

from copy import deepcopy

import pytest

from src.app.web.agent_request_support import normalize_debug_info
from src.core.contracts import DebugState, ResponseState
from src.eval.online_runner.response_parser import parse_agent_response
from src.runtime.agent_runtime.debug_collector import DebugCollector
from tests.web.answer_fixtures import cited_response


DECISIONS = [
    {"sequence": 1, "source": "add_user_message", "target": "planner", "reason": "below_watermark"},
    {"sequence": 2, "source": "planner", "target": "retrieve_dispatch", "reason": "retrieval_required:1_task(s)"},
    {"sequence": 3, "source": "post_synthesis_validation", "target": "synthesize", "reason": "unresolved_references"},
    {"sequence": 4, "source": "post_synthesis_validation", "target": "action_postprocess", "reason": "validation_passed"},
]
COMPACTIONS = [{"before": {"messages": 20}, "after": {"messages": 12},
                "removed_messages": 8, "summary_fallback": False}]


def collect_debug():
    answer = cited_response()
    state = {
        "response": ResponseState(result=answer, body_kind="compose"),
        "debug": DebugState(memory_compactions=deepcopy(COMPACTIONS)),
        "route_decisions": deepcopy(DECISIONS),
    }
    debug = DebugCollector().build(
        response=state, updated_messages=[], graph_total_ms=25, upload_retriever_build_ms=None,
    )
    return answer, state, debug


def test_collector_http_and_eval_preserve_decision_order_and_count():
    answer, state, debug = collect_debug()
    normalized = normalize_debug_info(debug, 30).model_dump(mode="json")
    parsed = parse_agent_response({"response": answer.model_dump(mode="json"), "debug": normalized})

    assert debug["route_decisions"] == DECISIONS
    assert normalized["route_decisions"] == DECISIONS
    assert [item.model_dump(mode="json") for item in parsed.route_decisions] == DECISIONS
    assert parsed.response_errors == []
    assert normalized["observability_status"] == "ok"
    assert debug["memory_compactions"] == COMPACTIONS
    assert normalized["memory_compactions"] == COMPACTIONS
    assert parsed.memory_compactions == COMPACTIONS
    assert state["route_decisions"] == DECISIONS
    assert "edge_decisions" not in debug
    assert "edge_decisions" not in normalized


@pytest.mark.parametrize("invalid", [None, {}, [{"source": "planner", "target": "retrieve_dispatch"}],
    [{**DECISIONS[0], "target": "unknown"}]])
def test_malformed_history_is_reported_by_http_and_eval(invalid):
    answer, _, debug = collect_debug()
    debug["route_decisions"] = invalid

    normalized = normalize_debug_info(debug, 30).model_dump(mode="json")
    parsed = parse_agent_response({"response": answer.model_dump(mode="json"), "debug": debug})

    assert normalized["observability_status"] == "failed"
    assert "route_decisions" in normalized["missing_required_debug_fields"]
    assert "DEBUG_NORMALIZATION_FAILED" in normalized["error_codes"]
    assert any("route_decisions" in error for error in normalized["errors"])
    assert any("route_decisions" in error for error in parsed.response_errors)
    assert parsed.debug_observability_status == "failed"


def test_missing_history_is_not_accepted_as_an_observed_empty_history():
    answer, state, debug = collect_debug()
    del debug["route_decisions"]
    normalized = normalize_debug_info(debug, 30).model_dump(mode="json")
    parsed = parse_agent_response({"response": answer.model_dump(mode="json"), "debug": debug})

    assert normalized["observability_status"] == "failed"
    assert "route_decisions" in normalized["missing_required_debug_fields"]
    assert parsed.debug_observability_status == "failed"
    assert "route_decisions" in parsed.missing_required_debug_fields

    del state["route_decisions"]
    collected = DebugCollector().build(response=state, updated_messages=[], graph_total_ms=0,
                                       upload_retriever_build_ms=None)
    assert collected["observability_status"] == "failed"
    assert "route_decisions" in collected["missing_required_debug_fields"]


def test_summary_fallback_remains_degraded_after_observation_boundaries():
    answer, state, _ = collect_debug()
    state["debug"] = state["debug"].model_copy(update={
        "observability_status": "degraded",
        "validation_events": ["memory_summary_fallback: reason=model_failed"],
        "memory_compactions": [{**COMPACTIONS[0], "summary_fallback": True}],
    })
    debug = DebugCollector().build(response=state, updated_messages=[], graph_total_ms=25,
                                   upload_retriever_build_ms=None)
    normalized = normalize_debug_info(debug, 30).model_dump(mode="json")
    parsed = parse_agent_response({"response": answer.model_dump(mode="json"), "debug": normalized})

    assert debug["observability_status"] == "degraded"
    assert normalized["observability_status"] == "degraded"
    assert parsed.debug_observability_status == "degraded"
    assert parsed.memory_compactions[0]["summary_fallback"] is True
    assert parsed.validation_events == ["memory_summary_fallback: reason=model_failed"]
