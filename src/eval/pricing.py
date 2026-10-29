from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from src.core.contracts.usage import LLMCallRecord

from .config_models import Pricing


def compute_cost_usd(
    *,
    llm_calls: Sequence[LLMCallRecord] | None,
    pricing: Pricing,
    cache_mode: Literal["observed", "uncached"] = "observed",
) -> float | None:
    """Price observed cache subsets, or compare all input at the uncached rate.

    Missing cache counters are not zeros. A counter is unnecessary only when
    its configured rate equals the base input rate. Prices must describe the
    actual endpoint, service tier and context length used by the caller.
    """
    if cache_mode not in {"observed", "uncached"}:
        raise ValueError("cache_mode must be observed or uncached")
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
        if cache_mode == "uncached":
            continue
        if usage.input_tokens is not None:
            observed_cache = sum(value for value in (usage.cached_input_tokens, usage.cache_write_input_tokens)
                                 if value is not None)
            if observed_cache > usage.input_tokens:
                return None
        for tokens, rate in (
            (usage.cached_input_tokens, model_pricing.cached_prompt_per_1k_usd),
            (usage.cache_write_input_tokens, model_pricing.cache_write_per_1k_usd),
        ):
            if tokens == 0 or rate == model_pricing.prompt_per_1k_usd:
                continue
            if tokens is None or rate is None:
                return None
            # Read/write counts are subsets of total input, never extra tokens.
            total_cost += (tokens / 1000.0) * (rate - model_pricing.prompt_per_1k_usd)
    return round(total_cost, 8)
