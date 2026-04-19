from __future__ import annotations

import logging
from typing import Any, List

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..contracts import GraphState
from ..contracts.boundary.debug import get_debug_state
from ..contracts.boundary.runtime import get_runtime_state
from ..contracts.debug import LLMCallMetadata, build_llm_call_metadata
from ..logging_utils import log_event

SUMMARY_SYS = (
    "Summarize the older conversation in 4-5 lines.\n"
    "- Keep topic, conclusions, decisions, key code/version/URL.\n"
    "- Remove duplication.\n"
    "- If uncertain, state uncertainty explicitly.\n"
)

logger = logging.getLogger(__name__)


def add_user_message(state: GraphState) -> GraphState:
    runtime = get_runtime_state(state)
    return {"messages": [HumanMessage(content=runtime.user_input)]}


def keep_recent_messages(messages: List[BaseMessage], max_turns: int = 6) -> List[BaseMessage]:
    if not messages:
        return messages
    turns_to_keep = max(0, int(max_turns)) + 1
    human_indices = [
        index for index, message in enumerate(messages) if isinstance(message, HumanMessage)
    ]
    if not human_indices or len(human_indices) <= turns_to_keep:
        return messages
    start_index = human_indices[-turns_to_keep]
    return messages[start_index:]


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


def _normalize_transcript_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _summary_role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    return "system"


def build_summary_transcript(messages: List[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        text = _normalize_transcript_text(extract_text_content(message.content))
        if not text:
            continue
        lines.append(f"{_summary_role(message)}: {text}")
    return "\n".join(lines)


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
    def summarize_old_messages(state: GraphState) -> GraphState:
        messages: List[BaseMessage] = state.get("messages", [])
        recent_window = keep_recent_messages(messages, max_turns=max_turns)
        if len(recent_window) == len(messages):
            return {}

        cutoff = len(messages) - len(recent_window)
        old_messages = messages[:cutoff]
        recent_messages = recent_window
        llm_calls: list[LLMCallMetadata] = []
        summary_transcript = build_summary_transcript(old_messages)

        if not summary_transcript:
            return {"messages": recent_messages}

        try:
            summary_response = llm_summarizer.invoke(
                [
                    SystemMessage(content=SUMMARY_SYS),
                    HumanMessage(content=summary_transcript),
                ]
            )
            summary = extract_text_content(getattr(summary_response, "content", summary_response)).strip()
            if isinstance(summary_response, AIMessage):
                llm_calls.append(
                    build_llm_call_metadata(
                        stage="summarize",
                        attempt=1,
                        path="direct",
                        message=summary_response,
                    )
                )
        except Exception as exc:
            if verbose:
                log_event(logger, logging.WARNING, "summary_failed", error=exc)
            return {"messages": recent_messages}

        runtime = get_runtime_state(state)
        debug = get_debug_state(state)
        previous_summary = (runtime.memory_summary or "").strip()
        merged_summary = (previous_summary + ("\n" if previous_summary else "") + summary).strip()
        updates: GraphState = {
            "runtime": runtime.model_copy(update={"memory_summary": merged_summary}),
            "messages": recent_messages,
        }
        if llm_calls:
            updates["debug"] = debug.model_copy(
                update={"llm_calls": [*debug.llm_calls, *llm_calls]}
            )
        if verbose:
            log_event(logger, logging.INFO, "summary_merged", cutoff=cutoff)
        return updates

    return summarize_old_messages
