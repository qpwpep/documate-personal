from __future__ import annotations

from collections.abc import Sequence

from src.core.contracts.usage import LLMCallRecord

from .config_models import ModelPricing, Pricing


def compute_cost_usd(
    *,
    llm_calls: Sequence[LLMCallRecord] | None,
    pricing: Pricing,
) -> float | None:
    """Price every observed call; an unavailable count makes the total unknown."""
    if llm_calls is None:
        return None
    total_cost = 0.0
    for call in llm_calls:
        usage = call.usage
        if usage.input_tokens is None or usage.output_tokens is None:
            return None
        model_pricing = _resolve_model_pricing(call.model_name, pricing)
        total_cost += (usage.input_tokens / 1000.0) * float(model_pricing.prompt_per_1k_usd)
        total_cost += (usage.output_tokens / 1000.0) * float(model_pricing.completion_per_1k_usd)
    return round(total_cost, 8)


def _resolve_model_pricing(model_name: str | None, pricing: Pricing) -> ModelPricing:
    if model_name:
        configured_pricing = pricing.models.get(str(model_name))
        if configured_pricing is not None:
            return configured_pricing
    return ModelPricing(
        prompt_per_1k_usd=float(pricing.prompt_per_1k_usd),
        completion_per_1k_usd=float(pricing.completion_per_1k_usd),
    )
