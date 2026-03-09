from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..settings import AppSettings
from .docs_search import build_docs_search_tool
from .local_rag import build_local_rag_tools
from .save_text import build_save_text_tool
from .slack_notify import build_slack_notify_tool


@dataclass(frozen=True)
class ToolRegistry:
    tavily_search_tool: Any
    rag_search_tool: Any
    upload_search_tool: Any
    save_text_tool: Any
    slack_notify_tool: Any
    all_tools: list[Any]


def build_tool_registry(settings: AppSettings) -> ToolRegistry:
    tavily_search_tool = build_docs_search_tool(settings)
    rag_search_tool, upload_search_tool = build_local_rag_tools(settings)
    save_text_tool = build_save_text_tool()
    slack_notify_tool = build_slack_notify_tool(settings)
    all_tools = [
        tavily_search_tool,
        rag_search_tool,
        upload_search_tool,
        save_text_tool,
        slack_notify_tool,
    ]
    return ToolRegistry(
        tavily_search_tool=tavily_search_tool,
        rag_search_tool=rag_search_tool,
        upload_search_tool=upload_search_tool,
        save_text_tool=save_text_tool,
        slack_notify_tool=slack_notify_tool,
        all_tools=all_tools,
    )
