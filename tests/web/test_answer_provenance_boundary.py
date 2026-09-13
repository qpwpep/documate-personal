from copy import deepcopy

from src.app.web.agent_request_support import normalize_debug_info
from src.core.contracts.boundary.debug import parse_debug_payload
from src.core.contracts.debug import DebugPayload
from tests.web.answer_fixtures import cited_response


def provenance_debug():
    response = cited_response()
    evidence = response.citations[0].evidence
    provenance = {
        "version": 1, "body_kind": "copy_answer", "response_hash": response.content_hash,
        "source": {"ref": "pending", "response_hash": response.content_hash,
                   "citation_ids": [evidence.id]},
        "evidence_packet": [evidence.model_dump(mode="json")],
    }
    return {**DebugPayload().model_dump(mode="json"), "answer_provenance": provenance,
            "tool_calls": ["save_text"], "tool_call_count": 1}


def test_graph_and_http_debug_preserve_captured_pending_source_and_exact_packet():
    """Both diagnostic boundaries retain the captured parent and source ranges without inventing searches."""
    raw = provenance_debug()
    expected = deepcopy(raw)

    graph_debug = parse_debug_payload(raw).model_dump(mode="json")
    http_debug = normalize_debug_info(graph_debug, 25).model_dump(mode="json")

    assert graph_debug["answer_provenance"] == expected["answer_provenance"]
    assert http_debug["answer_provenance"] == expected["answer_provenance"]
    assert http_debug["observed_hits"] == []
    assert http_debug["tool_calls"] == ["save_text"]
    assert http_debug["observability_status"] == "ok"
    assert raw == expected


def test_http_debug_marks_corrupt_packet_unavailable_without_losing_other_diagnostics():
    """A changed source under an old reference ID cannot survive normalization as valid provenance."""
    raw = provenance_debug()
    raw["answer_provenance"]["evidence_packet"][0]["element"]["text"] = "a different source"

    result = normalize_debug_info(raw, 25).model_dump(mode="json")

    assert result["answer_provenance"] is None
    assert result["observability_status"] == "failed"
    assert "answer_provenance" in result["missing_required_debug_fields"]
    assert "DEBUG_NORMALIZATION_FAILED" in result["error_codes"]
    assert result["tool_calls"] == ["save_text"]
    assert result["latency_ms_server"] == 25
