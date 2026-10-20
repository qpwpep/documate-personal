from __future__ import annotations

from collections.abc import Sequence

from src.core.contracts.usage import LLMCallRecord

from .config_models import Pricing


def compute_cost_usd(
    *,
    llm_calls: Sequence[LLMCallRecord] | None,
    pricing: Pricing,
) -> float | None:
    """Price each API path explicitly; unobserved billable usage stays unknown."""
    if llm_calls is None:
        return None
    total_cost = 0.0
    for call in llm_calls:
        usage = call.usage
        prices = pricing.decisions_models if call.path == "decisions" else pricing.models
        model_pricing = prices.get(call.model_name or "")
        if model_pricing is None:
            return None
        for tokens, rate in (
            (usage.input_tokens, model_pricing.prompt_per_1k_usd),
            (usage.output_tokens, model_pricing.completion_per_1k_usd),
        ):
            # Explicitly free dimensions need no usage estimate. Unknown rates
            # never reach here, and missing billable tokens never become zero.
            if rate == 0:
                continue
            if tokens is None:
                return None
            total_cost += (tokens / 1000.0) * float(rate)
    return round(total_cost, 8)
