from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from .planner_schema import PlannerOutput
from .settings import AppSettings


@dataclass(frozen=True)
class LLMRegistry:
    llm_planner: Any
    llm_synthesizer: Any
    llm_summarizer: Any
    verbose: bool


def build_llm_registry(settings: AppSettings) -> LLMRegistry:
    llm_synthesizer = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=settings.synthesis_max_tokens,
        timeout=settings.synthesis_timeout_seconds,
        max_retries=settings.synthesis_max_retries,
        verbose=settings.verbose,
    )

    llm_planner_base = ChatOpenAI(
        model=settings.planner_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=300,
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
        llm_summarizer=llm_summarizer,
        verbose=settings.verbose,
    )
