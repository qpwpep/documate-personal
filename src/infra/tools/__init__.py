from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.infra.settings import AppSettings
from src.infra.tools.docs_search import build_docs_search_tool
from src.infra.tools.local_rag import build_upload_search_tool
from src.infra.tools.save_text import build_save_text_tool
from src.infra.tools.slack_notify import build_slack_notify_tool


@dataclass(frozen=True)
class ToolRegistry:
    tavily_search_tool: Any
    upload_search_tool: Any
    save_text_tool: Any
    slack_notify_tool: Any
    all_tools: list[Any]


def build_tool_registry(settings: AppSettings) -> ToolRegistry:
    tavily_search_tool = build_docs_search_tool(settings)
    upload_search_tool = build_upload_search_tool(settings)
    save_text_tool = build_save_text_tool()
    slack_notify_tool = build_slack_notify_tool(settings)
    all_tools = [
        tavily_search_tool,
        upload_search_tool,
        save_text_tool,
        slack_notify_tool,
    ]
    return ToolRegistry(
        tavily_search_tool=tavily_search_tool,
        upload_search_tool=upload_search_tool,
        save_text_tool=save_text_tool,
        slack_notify_tool=slack_notify_tool,
        all_tools=all_tools,
    )
