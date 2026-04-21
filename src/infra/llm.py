from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

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
    return {"reasoning": {"effort": settings.synthesis_reasoning_effort}}


def build_llm_registry(settings: AppSettings) -> LLMRegistry:
    llm_synthesizer = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=settings.synthesis_max_tokens,
        timeout=settings.synthesis_timeout_seconds,
        max_retries=settings.synthesis_max_retries,
        use_responses_api=True,
        output_version="responses/v1",
        verbose=settings.verbose,
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
        use_responses_api=True,
        output_version="responses/v1",
        verbose=settings.verbose,
        **_build_synthesis_reasoning_kwargs(settings),
    )
    llm_planner_base = ChatOpenAI(
        model=settings.planner_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=settings.planner_max_tokens,
        timeout=30,
        max_retries=2,
        verbose=settings.verbose,
    )
    llm_planner = llm_planner_base.with_structured_output(
        PlannerOutput,
        method="json_schema",
        include_raw=True,
        strict=True,
    )

    llm_summarizer = ChatOpenAI(
        model=settings.summary_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=250,
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
