from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from .settings import AppSettings


@dataclass(frozen=True)
class LLMRegistry:
    llm_with_tools: Any
    llm_summarizer: Any
    verbose: bool


def build_llm_registry(settings: AppSettings, tools: list[Any]) -> LLMRegistry:
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
    )
    llm_with_tools = llm.bind_tools(tools)

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
        llm_with_tools=llm_with_tools,
        llm_summarizer=llm_summarizer,
        verbose=settings.verbose,
    )
