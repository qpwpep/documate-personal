"""Provider metadata travels through real runtime, SSE, evaluation and reporting."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from src.eval.config_models import BenchmarkCase, BenchmarkConfig, ModelPricing, Pricing
from src.eval.judge_llm import LLMJudge
from tests.eval.runner_helpers import run_case_with_weights as _run_single_case
from src.eval.reporting.summary import build_summary
from tests.web.test_agent_stream_integration import agent_server
from tests.web.test_multi_upload_api import LocalChatModel


class UsageModelBoundary(LocalChatModel):
    """Only the external provider is replaced; production nodes interpret the response."""

    def with_structured_output(self, schema, **kwargs):
        return UsageModelBoundary(self.controls, schema_name=schema["name"])

    def invoke(self, messages, **_kwargs):
        response = super().invoke(messages)
        if self.schema_name == "PlannerOutput":
            response["raw"] = AIMessage(content="", response_metadata={"model_name": "usage-planner"},
                usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
        elif self.schema_name == "AnswerDocument":
            # SDK message validation requires complete usage fields. Construct the
            # boundary response with the exact empty/partial metadata under test,
            # so the app's own normalizer (not a test fixture) makes the decision.
            response["raw"] = AIMessage(content="", response_metadata={
                "model_name": "usage-synthesis", "token_usage": self.controls.response_usage,
            }).model_copy(update={"usage_metadata": self.controls.usage_metadata})
        return response


@pytest.mark.parametrize(
    ("usage_metadata", "response_usage", "expected_input", "expected_output", "expected_cost", "conflict"),
    [
        ({}, {"prompt_tokens": 100, "completion_tokens": 20}, 100, 20, 0.826, False),
        ({"input_tokens": 0, "output_tokens": 0}, {"prompt_tokens": 100, "completion_tokens": 20}, 0, 0, 0.026, True),
        ({"output_tokens": 20}, {}, None, 20, None, False),
        ({"input_tokens": 100}, {}, 100, None, None, False),
    ],
    ids=["empty-primary-fallback", "authoritative-zero", "output-observed-input-missing", "input-observed-output-missing"],
)
def test_usage_contract_through_provider_runtime_sse_and_summary(
    agent_server, monkeypatch, usage_metadata, response_usage,
    expected_input, expected_output, expected_cost, conflict,
):
    endpoint, app, root = agent_server
    controls = SimpleNamespace(
        fail_synthesis=False, before_embedding=None, before_synthesis=None,
        planned_symbols=(), planned_file_ids=None,
        usage_metadata=usage_metadata, response_usage=response_usage,
    )
    monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **kwargs: UsageModelBoundary(controls))
    config = BenchmarkConfig(judge_enabled=False, pricing=Pricing(
        # Distinct per-model rates make incorrect aggregation before pricing observable.
        prompt_per_1k_usd=99, completion_per_1k_usd=99,
        models={
            "usage-planner": ModelPricing(prompt_per_1k_usd=1, completion_per_1k_usd=2),
            "usage-synthesis": ModelPricing(prompt_per_1k_usd=3, completion_per_1k_usd=5),
        },
    ))
    case = BenchmarkCase(
        case_id="usage-pipeline", category="tool_action", query="Write a brief response.",
        setup_turns=["Prepare a brief response."], setup_forbidden_tools=[[]],
    )
    result = _run_single_case(
        run_id="usage-pipeline", endpoint=endpoint, fixtures_path=root / "cases.jsonl",
        case=case, timeout_seconds=10, judge=LLMJudge(model_name="unused", enabled=False), config=config,
    )

    assert result.endpoint == endpoint + "/agent/stream"
    assert result.http_status == 200
    assert result.runtime_errors == result.response_errors == []
    assert result.response is not None
    assert app.state.session_store.active_session_ids() == {result.session_id}
    assert [turn.role for turn in result.scenario_turns] == ["setup", "question"]
    for turn in result.scenario_turns:
        assert turn.http_status == 200
        assert [call.stage for call in turn.llm_calls] == ["planner", "synthesis"]
        planner, synthesis = turn.llm_calls
        assert planner.model_name == "usage-planner"
        assert (planner.usage.input_tokens, planner.usage.output_tokens) == (7, 3)
        assert synthesis.model_name == "usage-synthesis"
        assert synthesis.usage.input_tokens == expected_input
        assert synthesis.usage.output_tokens == expected_output
        assert any(issue.code == "conflict" for issue in synthesis.usage.issues) is conflict
        # The actual HTTP debug carries only canonical calls, with raw metadata
        # retained as diagnostics. There is no independently interpreted total.
        wire_calls = turn.debug["llm_calls"]
        assert wire_calls[1]["usage"]["input_tokens"] == expected_input
        assert wire_calls[1]["usage"]["output_tokens"] == expected_output
        assert wire_calls[1]["usage_metadata"] == usage_metadata
        assert wire_calls[1]["response_metadata"]["token_usage"] == response_usage
        assert "token_usage" not in turn.debug
        assert "model_usage_status" not in turn.debug

    # The cost includes both turns and both models. Output includes only the
    # final question's synthesis, excluding both planners and setup synthesis.
    assert result.cost_usd == (pytest.approx(expected_cost) if expected_cost is not None else None)
    assert result.synthesis_output_tokens == expected_output
    summary = build_summary(
        run_id="usage-pipeline", endpoint=endpoint, fixtures_path="cases.jsonl", config_path="config.toml",
        track="smoke", requested_limit=None, config=config, cases=[case], results=[result],
    )
    assert summary.metrics.cost_observed_cases == (1 if expected_cost is not None else 0)
    assert summary.metrics.cost_observation_rate == (1.0 if expected_cost is not None else 0.0)
    assert summary.metrics.cost_gate_eligible is (expected_cost is not None)
