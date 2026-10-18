"""Decisions calls require a configured model price to report a known cost."""

from __future__ import annotations

import pytest

from src.core.contracts.usage import LLMCallRecord, TokenUsage
from src.eval.config_models import ModelPricing, Pricing
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
    pricing = Pricing(models={
        "gpt-6-luna": ModelPricing(prompt_per_1k_usd=0.01, completion_per_1k_usd=0.02),
    })

    assert compute_cost_usd(
        llm_calls=[_decision_call(), _decision_call("post_synthesis_validation")],
        pricing=pricing,
    ) == pytest.approx(0.024)


def test_confirmed_no_calls_still_have_zero_cost():
    assert compute_cost_usd(llm_calls=[], pricing=Pricing()) == 0.0
