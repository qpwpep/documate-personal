from __future__ import annotations

import logging
from typing import Any, List

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage

from ..logging_utils import log_event
from .state import State

SUMMARY_SYS = (
    "Summarize the older conversation in 4-5 lines.\n"
    "- Keep topic, conclusions, decisions, key code/version/URL.\n"
    "- Remove duplication.\n"
    "- If uncertain, state uncertainty explicitly.\n"
)

logger = logging.getLogger(__name__)


def add_user_message(state: State) -> State:
    messages = list(state.get("messages", []))
    messages.append(HumanMessage(content=state["user_input"]))
    state["messages"] = messages
    return state


def keep_recent_messages(messages: List[BaseMessage], max_turns: int = 6) -> List[BaseMessage]:
    if not messages:
        return messages
    window_size = max_turns * 2 + 2
    return messages[-window_size:]


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def latest_previous_ai_answer(messages: list[AnyMessage]) -> str:
    seen_current_user = False
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            if not seen_current_user:
                seen_current_user = True
                continue
            break
        if seen_current_user and isinstance(message, AIMessage):
            text = extract_text_content(message.content).strip()
            if text:
                return text
    return ""


def make_summarize_node(llm_summarizer: Any, verbose: bool, max_turns: int = 6):
    def summarize_old_messages(state: State) -> State:
        messages: List[BaseMessage] = state.get("messages", [])
        recent_window = keep_recent_messages(messages, max_turns=max_turns)
        if len(recent_window) == len(messages):
            return state

        cutoff = len(messages) - len(recent_window)
        old_messages = messages[:cutoff]
        recent_messages = messages[cutoff:]

        try:
            summary = llm_summarizer.invoke(
                [SystemMessage(content=SUMMARY_SYS)] + old_messages
            ).content.strip()
        except Exception as exc:
            if verbose:
                log_event(logger, logging.WARNING, "summary_failed", error=exc)
            state["messages"] = recent_messages
            return state

        previous_summary = (state.get("memory_summary") or "").strip()
        merged_summary = (previous_summary + ("\n" if previous_summary else "") + summary).strip()
        state["memory_summary"] = merged_summary
        state["messages"] = recent_messages
        if verbose:
            log_event(logger, logging.INFO, "summary_merged", cutoff=cutoff)
        return state

    return summarize_old_messages
