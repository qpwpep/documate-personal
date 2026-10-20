"""Judge input budgets remain observable in saved evaluation results."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.decisions import raw_results_fingerprint, results_fingerprint
from src.eval.io import load_config
from src.eval.judge_llm import LLMJudge
from src.eval.online_runner.response_parser import parse_agent_response
from src.eval.online_runner.result_builder import build_case_result
from src.eval.reporting import summary as summary_module
from src.eval.reporting.writer import load_run_outputs, write_run_outputs
from src.eval.result_models import CaseResult
from src.eval.weighting import resolve_case_weights
from tests.eval.test_release_eval_contract import _case_result, _config, _response_payload


def _summary(case, result, config):
    return summary_module.build_summary(
        run_id="judge-budget", endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="smoke", requested_limit=None,
        config=config, cases=[case], results=[result],
    )


def test_judge_budget_defaults_and_explicit_positive_limits():
    defaults = BenchmarkConfig()
    assert defaults.judge_max_input_tokens == 128_000
    assert defaults.judge_max_output_tokens == 4_096

    custom = BenchmarkConfig(judge_max_input_tokens=12_000, judge_max_output_tokens=1_024)
    assert custom.model_dump()["judge_max_input_tokens"] == 12_000
    assert custom.model_dump()["judge_max_output_tokens"] == 1_024


@pytest.mark.parametrize("field", ["judge_max_input_tokens", "judge_max_output_tokens"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1024", None])
def test_judge_budgets_require_strict_positive_integers(field, value):
    with pytest.raises(ValidationError):
        BenchmarkConfig(**{field: value})


@pytest.mark.parametrize("runtime, expected", [
    ('judge_model = "fixture-judge"\n', (128_000, 4_096)),
    ("judge_max_input_tokens = 12000\njudge_max_output_tokens = 1024\n", (12_000, 1_024)),
])
def test_load_config_reads_judge_budgets_from_runtime_table(tmp_path, runtime, expected):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[runtime]\n" + runtime, encoding="utf-8")

    config = load_config(config_path)

    assert (config.judge_max_input_tokens, config.judge_max_output_tokens) == expected


@pytest.mark.parametrize("field", ["judge_max_input_tokens", "judge_max_output_tokens"])
@pytest.mark.parametrize("value", ["0", '"1024"'])
def test_load_config_validates_runtime_judge_budget_values(tmp_path, field, value):
    config_path = tmp_path / "config.toml"
    config_path.write_text(f"[runtime]\n{field} = {value}\n", encoding="utf-8")

    with pytest.raises(ValidationError, match=field):
        load_config(config_path)


def test_load_config_still_rejects_unknown_runtime_keys_with_judge_budgets(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[runtime]\njudge_max_input_tokens = 12000\njudge_max_output_tokens = 1024\n"
        "unexpected_budget = 1000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown runtime config keys: unexpected_budget$"):
        load_config(config_path)


def test_unrecorded_judge_input_stats_do_not_change_saved_result_fingerprint():
    _, current = _case_result()
    saved = current.model_dump(mode="json")
    saved.pop("judge_input_stats", None)

    restored = CaseResult.model_validate(saved)

    assert restored.judge_input_stats is None
    assert "judge_input_stats" not in restored.model_dump(mode="json")
    assert results_fingerprint([restored]) == raw_results_fingerprint([saved])


def test_evaluation_fingerprint_identifies_judge_policy_and_budgets(monkeypatch):
    case, result = _case_result()
    config = _config()
    original = _summary(case, result, config).evaluation_fingerprint
    for field in ("judge_max_input_tokens", "judge_max_output_tokens"):
        changed = config.model_copy(update={field: getattr(config, field) + 1})
        assert _summary(case, result, changed).evaluation_fingerprint != original

    monkeypatch.setattr(summary_module, "JUDGE_INPUT_CONTRACT_VERSION", "future-input-policy")
    assert _summary(case, result, config).evaluation_fingerprint != original


def test_input_budget_failure_retains_stats_and_unknown_quality_on_roundtrip(tmp_path):
    case = BenchmarkCase(
        case_id="budget", category="docs_only", query="pandas merge는 어떻게 동작해?",
        expected_tools=["tavily_search"], forbidden_tools=["slack_notify"],
        require_official_citation=True,
    )
    config = _config().model_copy(update={"judge_max_input_tokens": 8})
    judge = LLMJudge(
        model_name="fixture-judge", enabled=False,
        max_input_tokens=config.judge_max_input_tokens,
        max_output_tokens=config.judge_max_output_tokens,
    )
    calls = []

    class ForbiddenJudgeBoundary:
        def invoke(self, messages):
            calls.append(messages)
            raise AssertionError("An oversized judge input must not reach the model.")

    judge.enabled = True
    judge.client = ForbiddenJudgeBoundary()
    result = build_case_result(
        run_id="judge-budget", endpoint_url="http://fixture", case=case, judge=judge,
        config=config, resolved_weights=resolve_case_weights(case=case, profiles=config.weights),
        session_id="budget-session", created_at="2026-01-01T00:00:00+00:00",
        request_payload={"query": case.query}, latency_ms_e2e=10,
        parsed_response=parse_agent_response(_response_payload()),
    )

    assert calls == []
    assert result.judge_status == "not_run"
    assert result.judge_status_reason == "input_budget_exceeded"
    assert result.judge_input_stats is not None
    assert result.eval_validity == "incomplete"
    assert result.composite_quality_score is None
    assert result.product_pass is None
    assert result.release_pass is False
    assert "product_quality_below_floor" not in result.gate_failures

    summary = _summary(case, result, config)
    write_run_outputs(output_dir=tmp_path, results=[result], summary=summary)
    restored_summary, restored_results = load_run_outputs(tmp_path)

    assert restored_results[0].judge_input_stats == result.judge_input_stats
    assert restored_results[0].judge_status_reason == "input_budget_exceeded"
    assert restored_results[0].product_pass is None
    assert restored_summary.evaluation_fingerprint == summary.evaluation_fingerprint
    assert restored_summary.overall_passed is False
