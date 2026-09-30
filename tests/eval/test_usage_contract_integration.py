"""Usage observations must mean the same thing through HTTP, pricing and reports."""

from types import SimpleNamespace

import pytest
import requests

from src.core.contracts.debug import DebugPayload, build_llm_call_metadata
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.io import dump_jsonl
from src.eval.online_runner import run_online_benchmark
from src.eval.reporting import build_summary
from src.eval.reporting.writer import load_run_outputs
from tests.eval.response_fixtures import (
    answer_provenance, canonical_llm_call, execution_evidence, plain_response, sse_http_response,
)
from tests.eval.test_runner_sse import final_payload, run_case


pytestmark = pytest.mark.usefixtures("empty_upload_manifest_http")


@pytest.mark.parametrize(("usage_metadata", "response_usage", "cost", "output"), [
    ({}, {"prompt_tokens": 100, "completion_tokens": 20}, 0.000027, 20),
    ({"input_tokens": 0, "output_tokens": 0}, {"prompt_tokens": 100, "completion_tokens": 20}, 0.0, 0),
    ({"input_tokens": 100}, {}, None, None),
    ({"output_tokens": 20}, {}, None, 20),
], ids=["empty-primary-falls-back", "observed-zero-is-not-missing", "partial-is-not-zero", "output-observed-without-input"])
def test_usage_observation_reaches_cost_output_and_summary(
    usage_metadata, response_usage, cost, output, monkeypatch,
):
    message = SimpleNamespace(
        usage_metadata=usage_metadata,
        response_metadata={"token_usage": response_usage},
    )
    call = build_llm_call_metadata(stage="synthesis", attempt=1, path="structured", message=message)
    payload = final_payload()
    payload["debug"]["llm_calls"] = [call.model_dump(mode="json")]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: sse_http_response(200, payload))

    result = run_case()
    assert result.cost_usd == (pytest.approx(cost) if cost is not None else None)
    assert result.synthesis_output_tokens == output
    summary = build_summary(
        run_id="run-sse", endpoint="http://127.0.0.1:8000/", fixtures_path="cases.jsonl",
        config_path="config.toml", track="smoke", requested_limit=None,
        config=BenchmarkConfig(judge_enabled=False),
        cases=[BenchmarkCase(case_id="stream-case", category="tool_action", query="예제 저장")],
        results=[result],
    )
    assert summary.metrics.cost_observation_rate == (1.0 if cost is not None else 0.0)


@pytest.mark.parametrize(("setup_calls", "question_calls", "cost", "output", "pricing"), [
    ([canonical_llm_call(100, 20)], [
        canonical_llm_call(10, 2, stage="summarize", path="direct"),
        canonical_llm_call(10, 2, stage="planner"),
        canonical_llm_call(10, 2),
        canonical_llm_call(10, 3, attempt=2, path="structured_compact_fallback"),
    ], 0.0000384, 5, None),
    ([canonical_llm_call(100, 20)], [], 0.000027, 0, None),
    ([canonical_llm_call()], [canonical_llm_call(100, 20)], None, 20, None),
    ([], [canonical_llm_call(), canonical_llm_call(100, 20, attempt=2)], None, None, None),
    ([], [canonical_llm_call(None, 0), canonical_llm_call(100, 20, attempt=2)], None, 20, None),
    ([], [], 0.0, 0, None),
    ([], None, None, None, None),
    ([canonical_llm_call(1, 0)], [canonical_llm_call(1, 0)], 0.00000001, 0,
     {"prompt_per_1k_usd": 0.0000026, "completion_per_1k_usd": 0.0}),
], ids=["stage-and-turn-scopes", "deterministic-question", "missing-setup-usage", "missing-first-attempt",
        "partial-input-still-has-output", "confirmed-no-calls", "call-list-unavailable", "round-scenario-total-once"])
def test_scenario_usage_scope_survives_http_storage_and_report(
    setup_calls, question_calls, cost, output, pricing, tmp_path, monkeypatch,
):
    """Every cost uses all turns, while output uses only final-question synthesis attempts."""
    def post(url, *, json, **kwargs):
        response = plain_response("Observed answer")
        calls = setup_calls if json["query"] == "prepare" else question_calls
        request_id = json["query"]
        debug = DebugPayload(
            llm_calls=calls,
            execution_evidence=execution_evidence(request_id=request_id),
            answer_provenance=answer_provenance(response),
        ).model_dump(mode="json")
        # Unrelated legacy-looking diagnostics must never outrank canonical calls.
        debug["token_usage"] = {"prompt_tokens": 99999, "completion_tokens": 99999}
        debug["model_usage_status"] = "deterministic"
        return sse_http_response(200, {
            "response": response, "debug": debug, "trace": f"Request ID: {request_id}",
            "upload_manifest": {"epoch": "fixture-epoch", "revision": 0, "files": []},
        })

    monkeypatch.setattr(requests, "post", post)
    fixtures = tmp_path / "cases.jsonl"
    dump_jsonl(fixtures, [BenchmarkCase(
        case_id="usage", category="tool_action", query="question", setup_turns=["prepare"],
        setup_forbidden_tools=[[]],
    )])
    output_dir, results, summary = run_online_benchmark(
        fixtures_path=fixtures, endpoint="http://fixture",
        config=BenchmarkConfig(judge_enabled=False, **({"pricing": pricing} if pricing else {})),
        config_path=tmp_path / "config.toml", output_root=tmp_path / "results", track="smoke",
    )
    result = results[0]
    assert result.runtime_errors == result.response_errors == []
    assert [turn.role for turn in result.scenario_turns] == ["setup", "question"]
    assert result.cost_usd == (pytest.approx(cost) if cost is not None else None)
    assert result.synthesis_output_tokens == output
    assert summary.metrics.cost_observation_rate == (1.0 if cost is not None else 0.0)
    assert summary.metrics.cost_observed_cases == int(cost is not None)
    assert summary.metrics.avg_cost_per_case_usd == (pytest.approx(cost) if cost is not None else None)
    assert summary.measurement_contract_version == "llm-usage-scenario-v2"
    stored_summary, stored_results = load_run_outputs(output_dir)
    assert stored_results[0].model_dump() == result.model_dump()
    assert stored_summary.model_dump() == summary.model_dump()
    assert "token_usage" not in result.model_dump()
    assert "llm_calls" not in result.model_dump()
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "cost_observation_rate" in report
    assert "llm_call_coverage_rate" not in report


def test_raw_only_call_cannot_rescue_a_malformed_canonical_call(monkeypatch):
    """A malformed entry keeps the scope unknown instead of pricing a surviving partial list."""
    payload = final_payload()
    payload["debug"]["llm_calls"] = [canonical_llm_call(100, 20), {
        "stage": "synthesis", "attempt": 2, "path": "structured_compact_fallback",
        "usage_metadata": {"input_tokens": 100, "output_tokens": 20},
    }]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: sse_http_response(200, payload))
    result = run_case()
    assert result.cost_usd is None
    assert result.synthesis_output_tokens is None
    assert result.scenario_turns[0].llm_calls is None
    assert any("llm_calls[1]" in error for error in result.response_errors)
