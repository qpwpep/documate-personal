from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from openai.lib._pydantic import to_strict_json_schema

from src.core.planner_schema import PlannerOutput
from src.infra.settings import AppSettings


@dataclass(frozen=True)
class LLMRegistry:
    llm_planner: Any
    llm_synthesizer: Any
    llm_synthesizer_compact: Any | None
    llm_summarizer: Any
    verbose: bool


def _derive_compact_synthesis_profile(settings: AppSettings) -> tuple[int, int]:
    return (
        max(1, int(settings.synthesis_max_tokens) // 2),
        max(1, int(settings.synthesis_timeout_seconds) // 2),
    )


def _build_synthesis_reasoning_kwargs(settings: AppSettings) -> dict[str, Any]:
    if not settings.synthesis_reasoning_effort:
        return {}
    if not settings.synthesis_use_responses_api:
        return {"reasoning_effort": settings.synthesis_reasoning_effort}
    return {"reasoning": {"effort": settings.synthesis_reasoning_effort}}


def _build_synthesis_api_kwargs(settings: AppSettings) -> dict[str, Any]:
    if not settings.synthesis_use_responses_api:
        return {"use_responses_api": False}
    return {
        "use_responses_api": True,
        "output_version": "responses/v1",
    }


def _build_planner_response_schema() -> dict[str, Any]:
    return {
        "name": "PlannerOutput",
        "strict": True,
        "schema": to_strict_json_schema(PlannerOutput),
    }


def build_llm_registry(settings: AppSettings) -> LLMRegistry:
    llm_synthesizer = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=settings.synthesis_max_tokens,
        timeout=settings.synthesis_timeout_seconds,
        max_retries=settings.synthesis_max_retries,
        verbose=settings.verbose,
        **_build_synthesis_api_kwargs(settings),
        **_build_synthesis_reasoning_kwargs(settings),
    )
    compact_max_tokens, compact_timeout = _derive_compact_synthesis_profile(settings)
    llm_synthesizer_compact = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=compact_max_tokens,
        timeout=compact_timeout,
        max_retries=0,
        verbose=settings.verbose,
        **_build_synthesis_api_kwargs(settings),
        **_build_synthesis_reasoning_kwargs(settings),
    )
    llm_planner_base = ChatOpenAI(
        model=settings.planner_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=settings.planner_max_tokens,
        timeout=30,
        max_retries=0 if settings.planner_hedge_delay_seconds > 0 else 2,
        verbose=settings.verbose,
    )
    llm_planner = llm_planner_base.with_structured_output(
        _build_planner_response_schema(),
        method="json_schema",
        include_raw=True,
        strict=True,
    )

    llm_summarizer = ChatOpenAI(
        model=settings.summary_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=settings.memory_summary_max_tokens,
        timeout=60,
        max_retries=2,
        verbose=settings.verbose,
    )

    return LLMRegistry(
        llm_planner=llm_planner,
        llm_synthesizer=llm_synthesizer,
        llm_synthesizer_compact=llm_synthesizer_compact,
        llm_summarizer=llm_summarizer,
        verbose=settings.verbose,
    )
