"""Malformed diagnostic fields fail one case without losing usable usage or later results."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests

from src.core.contracts.debug import DebugPayload, LLMCallMetadata, TokenUsage
from src.app.web.agent_request_support import normalize_debug_info
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.io import dump_jsonl
from src.eval.judge_llm import LLMJudge
from src.eval.main import command_report
from src.eval.online_runner import _run_single_case, run_online_benchmark
from tests.eval.response_fixtures import answer_provenance, execution_evidence, plain_response, sse_http_response


pytestmark = pytest.mark.usefixtures("empty_upload_manifest_http")


@pytest.mark.parametrize("setup", [False, True], ids=["final", "preparation"])
def test_invalid_retrieval_observation_cannot_erase_a_forbidden_start(setup, monkeypatch, tmp_path):
    """Malformed diagnostics retain the separate, positive evidence of a prohibited invocation."""
    payload = response_payload()
    payload["debug"]["retrieval_diagnostics"] = [{"status": []}]
    payload["debug"]["execution_evidence"] = execution_evidence(["upload_search"], request_id="diagnostic-request")
    payload["debug"]["tool_calls"] = ["upload_search"]
    payload["debug"]["tool_call_count"] = 1
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: sse_http_response(200, payload))
    result = _run_single_case(
        run_id="diagnostics", endpoint="http://fixture", fixtures_path=tmp_path / "cases.jsonl",
        case=BenchmarkCase(case_id="invalid", category="tool_action", query="final",
                           setup_turns=["prepare"] if setup else [],
                           setup_forbidden_tools=[["upload_search"]] if setup else None,
                           forbidden_tools=[] if setup else ["upload_search"]),
        timeout_seconds=1, judge=LLMJudge(model_name="unused", enabled=False),
        config=BenchmarkConfig(judge_enabled=False),
    )
    assert result.policy_assessment.status == "violated"
    assert "forbidden_tool_execution" in result.policy_assessment.failure_codes
    assert "tool_execution_retrieval_diagnostics_invalid" in result.policy_assessment.failure_codes
    assert not result.release_pass


@pytest.mark.parametrize("setup", [False, True], ids=["final", "preparation"])
def test_server_normalized_invalid_retrieval_observation_remains_unverified(setup, monkeypatch, tmp_path):
    """Server normalization preserves its rejection across the HTTP and policy boundaries."""
    payload = response_payload()
    payload["debug"]["retrieval_diagnostics"] = None
    payload["debug"] = normalize_debug_info(payload["debug"], latency_ms_server=1).model_dump(mode="json")
    assert payload["debug"]["retrieval_diagnostics"] == []
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: sse_http_response(200, payload))
    result = _run_single_case(
        run_id="diagnostics", endpoint="http://fixture", fixtures_path=tmp_path / "cases.jsonl",
        case=BenchmarkCase(case_id="invalid", category="tool_action", query="final",
                           setup_turns=["prepare"] if setup else [],
                           setup_forbidden_tools=[[]] if setup else None),
        timeout_seconds=1, judge=LLMJudge(model_name="unused", enabled=False),
        config=BenchmarkConfig(judge_enabled=False),
    )
    assert result.policy_assessment.status == "indeterminate"
    assert "tool_execution_retrieval_diagnostics_invalid" in result.policy_assessment.failure_codes
    assert any("retrieval_diagnostics" in error for error in result.response_errors)
    assert not result.release_pass


def response_payload():
    debug = DebugPayload(
        execution_evidence=execution_evidence(request_id="diagnostic-request"),
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


def test_malformed_tool_calls_keeps_usage_and_writes_the_following_case_report(tmp_path, monkeypatch):
    """A scalar tool list is a response contract failure while both cases reach persisted reports."""
    payloads = []

    def post(url, *, json, **kwargs):
        payload = response_payload()
        if json["query"] == "broken diagnostics":
            payload["debug"]["tool_calls"] = 1
        payloads.append(json)
        return sse_http_response(200, payload)

    monkeypatch.setattr(requests, "post", post)
    fixtures = tmp_path / "cases.jsonl"
    dump_jsonl(fixtures, [
        BenchmarkCase(case_id="broken", category="tool_action", query="broken diagnostics"),
        BenchmarkCase(case_id="following", category="tool_action", query="valid diagnostics"),
    ])

    output, results, summary = run_online_benchmark(
        fixtures_path=fixtures, endpoint="http://fixture", config=BenchmarkConfig(judge_enabled=False),
        config_path=tmp_path / "config.toml", output_root=tmp_path / "results", track="smoke",
    )

    assert [item["query"] for item in payloads] == ["broken diagnostics", "valid diagnostics"]
    first, following = results
    assert first.response is not None
    assert first.runtime_errors == []
    assert any("debug.tool_calls" in error for error in first.response_errors)
    assert first.debug["tool_calls"] == first.scenario_turns[0].debug["tool_calls"] == 1
    assert first.token_usage == TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    assert first.cost_usd == pytest.approx(0.0000027)
    assert not first.release_pass
    assert following.runtime_errors == following.response_errors == []
    assert summary.metrics.total_cases == 2
    stored = [json.loads(line) for line in (output / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [item["case_id"] for item in stored] == ["broken", "following"]
    assert stored[0]["debug"]["tool_calls"] == 1
    assert (output / "report.md").is_file()


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


@pytest.mark.parametrize(("defect", "status"), [
    ("missing", "unavailable"),
    ("missing_packet", "invalid"),
    ("wrong_hash", "invalid"),
])
def test_online_rejects_unproven_inputs_without_losing_the_answer_or_usage(defect, status, monkeypatch, tmp_path):
    """Missing or malformed provenance fails evaluation while preserving the actual response and billable usage."""
    payload = response_payload()
    if defect == "missing":
        payload["debug"].pop("answer_provenance")
    elif defect == "missing_packet":
        payload["debug"]["answer_provenance"].pop("evidence_packet")
    else:
        payload["debug"]["answer_provenance"]["response_hash"] = "another-answer"
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: sse_http_response(200, payload))

    result = _run_single_case(
        run_id="provenance", endpoint="http://fixture", fixtures_path=tmp_path / "cases.jsonl",
        case=BenchmarkCase(case_id="invalid", category="tool_action", query="test provenance"),
        timeout_seconds=1, judge=LLMJudge(model_name="unused", enabled=False),
        config=BenchmarkConfig(judge_enabled=False),
    )

    assert result.response.model_dump(mode="json") == payload["response"]
    assert result.debug == result.scenario_turns[0].debug == payload["debug"]
    assert result.runtime_errors == []
    assert result.evidence_assessment.status == status
    assert result.response_errors
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
