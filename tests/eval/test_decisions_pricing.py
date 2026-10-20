"""Every API path needs an explicit, valid model price for a known cost."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.contracts.usage import LLMCallRecord, TokenUsage
from src.eval.config_models import ModelPricing, Pricing
from src.eval.io import load_config
from src.eval.pricing import compute_cost_usd


def _decision_call(stage="pre_synthesis_validation", *, model="gpt-6-luna"):
    return LLMCallRecord(
        stage=stage, path="decisions", model_name=model,
        usage=TokenUsage(input_tokens=1000, output_tokens=100),
    )


def test_unknown_decisions_price_prevents_partially_priced_total():
    calls = [
        LLMCallRecord(
            stage="synthesis", path="structured", model_name="known-model",
            usage=TokenUsage(input_tokens=1000, output_tokens=100),
        ),
        _decision_call(),
    ]
    pricing = Pricing(models={
        "known-model": ModelPricing(prompt_per_1k_usd=0.01, completion_per_1k_usd=0.02),
    })

    assert compute_cost_usd(llm_calls=calls, pricing=pricing) is None


def test_configured_decisions_price_is_included_in_total():
    pricing = Pricing(decisions_models={
        "gpt-6-luna": ModelPricing(prompt_per_1k_usd=0.01, completion_per_1k_usd=0.02),
    })

    assert compute_cost_usd(
        llm_calls=[_decision_call(), _decision_call("post_synthesis_validation")],
        pricing=pricing,
    ) == pytest.approx(0.024)


def test_confirmed_no_calls_still_have_zero_cost():
    assert compute_cost_usd(llm_calls=[], pricing=Pricing()) == 0.0


def test_same_model_uses_the_price_for_its_actual_api_path():
    pricing = Pricing(
        models={"shared-model": ModelPricing(prompt_per_1k_usd=0.02, completion_per_1k_usd=0.04)},
        decisions_models={"shared-model": ModelPricing(prompt_per_1k_usd=0.01, completion_per_1k_usd=0)},
    )
    calls = [
        _decision_call(model="shared-model"),
        LLMCallRecord(
            stage="synthesis", path="structured", model_name="shared-model",
            usage=TokenUsage(input_tokens=1000, output_tokens=100),
        ),
    ]

    assert compute_cost_usd(llm_calls=calls, pricing=pricing) == pytest.approx(0.034)


def test_general_model_price_does_not_substitute_for_unknown_decisions_price():
    pricing = Pricing(models={
        "gpt-6-luna": ModelPricing(prompt_per_1k_usd=0.01, completion_per_1k_usd=0.02),
    })

    assert compute_cost_usd(llm_calls=[_decision_call()], pricing=pricing) is None


@pytest.mark.parametrize("model", ["unregistered-model", None])
def test_unknown_general_model_price_never_becomes_an_observed_cost(model):
    call = LLMCallRecord(
        stage="synthesis", path="structured", model_name=model,
        usage=TokenUsage(input_tokens=1000, output_tokens=100),
    )

    assert compute_cost_usd(llm_calls=[call], pricing=Pricing()) is None


def test_decisions_price_does_not_substitute_for_unknown_general_model_price():
    pricing = Pricing(decisions_models={
        "shared-model": ModelPricing(prompt_per_1k_usd=0.0001, completion_per_1k_usd=0),
    })
    call = LLMCallRecord(
        stage="synthesis", path="structured", model_name="shared-model",
        usage=TokenUsage(input_tokens=1000, output_tokens=100),
    )

    assert compute_cost_usd(llm_calls=[call], pricing=pricing) is None


@pytest.mark.parametrize("field", ["prompt_per_1k_usd", "completion_per_1k_usd"])
@pytest.mark.parametrize("value", [-0.02, float("nan"), float("inf"), float("-inf")])
def test_model_prices_must_be_finite_nonnegative_numbers(field, value):
    values = {"prompt_per_1k_usd": 0.01, "completion_per_1k_usd": 0.02, field: value}

    with pytest.raises(ValidationError):
        ModelPricing.model_validate(values)


def test_unscoped_fallback_prices_are_rejected():
    with pytest.raises(ValidationError):
        Pricing.model_validate({"prompt_per_1k_usd": 0.0001, "completion_per_1k_usd": 0.0002})


@pytest.mark.parametrize("table", ["models", "decisions_models"])
@pytest.mark.parametrize("name", ["", "   "])
def test_price_tables_require_explicit_model_names(table, name):
    with pytest.raises(ValidationError):
        Pricing.model_validate({table: {name: {
            "prompt_per_1k_usd": 0.0, "completion_per_1k_usd": 0.0,
        }}})


def test_free_output_does_not_require_an_output_token_observation():
    pricing = Pricing(decisions_models={
        "gpt-6-luna": ModelPricing(prompt_per_1k_usd=0.0001, completion_per_1k_usd=0),
    })
    call = LLMCallRecord(
        stage="pre_synthesis_validation", path="decisions", model_name="gpt-6-luna",
        usage=TokenUsage(input_tokens=1000, output_tokens=None),
    )

    assert compute_cost_usd(llm_calls=[call], pricing=pricing) == pytest.approx(0.0001)


def test_missing_billable_input_stays_unknown():
    pricing = Pricing(decisions_models={
        "gpt-6-luna": ModelPricing(prompt_per_1k_usd=0.0001, completion_per_1k_usd=0),
    })
    call = LLMCallRecord(
        stage="pre_synthesis_validation", path="decisions", model_name="gpt-6-luna",
        usage=TokenUsage(input_tokens=None, output_tokens=0),
    )

    assert compute_cost_usd(llm_calls=[call], pricing=pricing) is None


def test_shipped_config_prices_the_product_decisions_model():
    config = load_config(Path(__file__).resolve().parents[2] / "data/benchmarks/config.toml")

    assert compute_cost_usd(
        llm_calls=[_decision_call(), _decision_call("post_synthesis_validation")],
        pricing=config.pricing,
    ) == pytest.approx(0.0002)
