from __future__ import annotations

from typing import Any

from .config_models import ModelPricing, Pricing


def compute_cost_usd(
    *,
    token_usage: Any,
    llm_calls: list[Any] | None = None,
    pricing: Pricing,
) -> float | None:
    if llm_calls:
        total_cost = 0.0
        has_usage = False
        for call in llm_calls:
            prompt_tokens, completion_tokens = _extract_usage_from_llm_call(call)
            if prompt_tokens <= 0 and completion_tokens <= 0:
                continue
            has_usage = True
            model_pricing = _resolve_model_pricing(_extract_model_name_from_llm_call(call), pricing)
            total_cost += (prompt_tokens / 1000.0) * float(model_pricing.prompt_per_1k_usd)
            total_cost += (completion_tokens / 1000.0) * float(model_pricing.completion_per_1k_usd)
        if has_usage:
            return round(total_cost, 8)

    if token_usage is None:
        return None
    prompt_tokens = int(getattr(token_usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(token_usage, "completion_tokens", 0) or 0)
    prompt_cost = (prompt_tokens / 1000.0) * float(pricing.prompt_per_1k_usd)
    completion_cost = (completion_tokens / 1000.0) * float(pricing.completion_per_1k_usd)
    return round(prompt_cost + completion_cost, 8)


def _resolve_model_pricing(model_name: str | None, pricing: Pricing) -> ModelPricing:
    if model_name:
        configured_pricing = pricing.models.get(str(model_name))
        if configured_pricing is not None:
            return configured_pricing
    return ModelPricing(
        prompt_per_1k_usd=float(pricing.prompt_per_1k_usd),
        completion_per_1k_usd=float(pricing.completion_per_1k_usd),
    )


def _extract_usage_from_llm_call(llm_call: Any) -> tuple[int, int]:
    if not isinstance(llm_call, dict):
        return 0, 0

    usage_metadata = llm_call.get("usage_metadata")
    response_metadata = llm_call.get("response_metadata")
    usage_candidates = []
    if isinstance(usage_metadata, dict):
        usage_candidates.append(usage_metadata)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            usage_candidates.append(token_usage)

    for usage in usage_candidates:
        try:
            prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            completion_tokens = int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )
        except (TypeError, ValueError):
            continue
        return max(0, prompt_tokens), max(0, completion_tokens)

    return 0, 0


def _extract_model_name_from_llm_call(llm_call: Any) -> str | None:
    if not isinstance(llm_call, dict):
        return None
    response_metadata = llm_call.get("response_metadata")
    if not isinstance(response_metadata, dict):
        return None
    model_name = response_metadata.get("model_name") or response_metadata.get("model")
    return str(model_name) if model_name else None
