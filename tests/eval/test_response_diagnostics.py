"""Malformed diagnostic fields fail one case without losing usable usage or later results."""

from __future__ import annotations

import json

import pytest
import requests

from src.core.contracts.debug import DebugPayload, LLMCallMetadata, TokenUsage
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.io import dump_jsonl
from src.eval.judge_llm import LLMJudge
from src.eval.online_runner import _run_single_case, run_online_benchmark
from tests.eval.response_fixtures import answer_provenance, plain_response, sse_http_response




def response_payload():
    debug = DebugPayload(
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        llm_calls=[LLMCallMetadata(
            stage="planner", path="structured",
            response_metadata={"model_name": "local-test-model"},
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )],
        models_used=["local-test-model"], model_usage_status="llm_used",
    ).model_dump(mode="json")
    response = plain_response("Retained answer")
    debug["answer_provenance"] = answer_provenance(response)
    return {
        "response": response, "debug": debug,
        "trace": "Request ID: diagnostic-request",
        "upload_manifest": {"epoch": "fixture-epoch", "revision": 0, "files": []},
    }




@pytest.mark.parametrize(("field", "value"), [
    ("tool_calls", None),
    ("tool_calls", "upload_search"),
    ("tool_calls", [{"name": "upload_search"}]),
    ("schema_version", float("inf")),
    ("tool_call_count", float("inf")),
    ("latency_ms_server", float("inf")),
    ("token_usage", {"prompt_tokens": float("inf")}),
    ("llm_calls", [{"stage": "planner", "path": "structured", "attempt": float("inf")}]),
    ("llm_calls", [{"stage": "planner", "path": "structured", "usage_metadata": {"input_tokens": float("inf")}}]),
    ("llm_calls", [{"stage": "planner", "path": "structured", "response_metadata": {"token_usage": {"completion_tokens": float("inf")}}}]),
    ("retrieval_diagnostics", [{"attempt": float("inf")}]),
    ("retrieval_diagnostics", [{"answerability": {"invalid": True}}]),
    ("planner_diagnostics", {"override_reason": []}),
])
def test_invalid_diagnostic_value_keeps_other_fields_and_finishes_the_case(field, value, monkeypatch, tmp_path):
    """Malformed numbers and nested diagnostics are reported without crashing result collection."""
    payload = response_payload()
    payload["debug"][field] = value
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: sse_http_response(200, payload))

    result = _run_single_case(
        run_id="diagnostics", endpoint="http://fixture", fixtures_path=tmp_path / "cases.jsonl",
        case=BenchmarkCase(case_id="invalid", category="tool_action", query="test diagnostics"),
        timeout_seconds=1, judge=LLMJudge(model_name="unused", enabled=False),
        config=BenchmarkConfig(judge_enabled=False),
    )

    assert result.runtime_errors == []
    assert any(field in error for error in result.response_errors)
    assert result.response is not None
    assert result.debug[field] == value
    assert result.cost_usd == pytest.approx(0.0000027)
    assert not result.release_pass


def test_invalid_trace_preserves_the_answer_and_diagnostic_usage(monkeypatch, tmp_path):
    """A malformed trace cannot discard a valid answer or prevent scenario result serialization."""
    payload = response_payload()
    payload["trace"] = ["invalid trace"]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: sse_http_response(200, payload))

    result = _run_single_case(
        run_id="diagnostics", endpoint="http://fixture", fixtures_path=tmp_path / "cases.jsonl",
        case=BenchmarkCase(case_id="invalid", category="tool_action", query="test diagnostics"),
        timeout_seconds=1, judge=LLMJudge(model_name="unused", enabled=False),
        config=BenchmarkConfig(judge_enabled=False),
    )

    assert result.runtime_errors == []
    assert result.response_errors == ["trace must be a string"]
    assert result.trace is None
    assert result.response is not None
    assert result.debug == payload["debug"]
    assert result.cost_usd == pytest.approx(0.0000027)
    assert not result.release_pass




def test_result_builder_rejects_a_declared_provenance_mismatch_without_citation_requirements():
    """Any explicitly supplied invalid provenance prevents release success, including source-free action cases."""
    from src.eval.online_runner.response_parser import parse_agent_response
    from src.eval.online_runner.result_builder import build_case_result

    payload = response_payload()
    payload["debug"]["answer_provenance"]["response_hash"] = "different-answer"
    result = build_case_result(
        run_id="provenance", endpoint_url="http://fixture/agent/stream",
        case=BenchmarkCase(case_id="mismatch", category="tool_action", query="test"),
        judge=LLMJudge(model_name="unused", enabled=False), config=BenchmarkConfig(judge_enabled=False),
        session_id="scenario", created_at="2026-01-01T00:00:00+00:00", request_payload={},
        latency_ms_e2e=1, parsed_response=parse_agent_response(payload),
    )

    assert result.evidence_assessment.status == "invalid"
    assert any("response_hash" in error for error in result.response_errors)
    assert not result.release_pass
