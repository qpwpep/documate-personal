from __future__ import annotations

import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.conversation_memory import build_untrusted_memory_prompt_messages
from src.core.contracts import GraphState
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.runtime.nodes.retry import format_retry_context_for_planner
from src.runtime.nodes.session import keep_recent_messages


PLANNER_SYS = (
    "You are a retrieval planner. Return a structured plan only.\n"
    "Rules:\n"
    "- Choose retrieval routes from: docs, upload.\n"
    "- docs: official/latest docs on the web.\n"
    "- upload: currently uploaded-file retriever context.\n"
    "- If retrieval is unnecessary, set use_retrieval=false and tasks=[].\n"
    "- If retrieval is needed, set use_retrieval=true and include 1-2 tasks.\n"
    "- Each selected route must appear at most once.\n"
    "- Keep each task.query short and route-specific.\n"
    "- For docs tasks, preserve the library/framework name in task.query, even for bare library-level requests.\n"
    "- If the request is only asking to save/share/send the current answer, retrieval is unnecessary.\n"
    "- Plan the sources the answer requires, independently of tool or file availability. Include upload even when the referenced file has not been provided; the executor will request the missing file.\n"
    "- Distinguish a technical topic from evidence to inspect: describing a file format or an API does not require the user's files, while reporting what their code or notebook contains does.\n"
    "- General questions about file operations, upload APIs, file formats, or a future project are docs topics and do not require a user file.\n"
    "- Resolve references, negation, scope, and later corrections across the whole request. Omit any excluded source, whether docs or upload. A source named only to exclude it is not a requested source.\n"
    "- For search queries preserve the actual subject, identifiers, Korean terms, and comparison targets; omit delivery instructions and source-exclusion wording.\n"
    "- UploadSearch can search only the current uploaded file, not an entire project or a separate notebook index.\n"
    "- If the user asks only about the currently uploaded file/code, choose upload only; do not add docs unless official/current/latest documentation is explicitly requested.\n"
    "- For official docs plus file comparisons, choose docs and upload.\n"
    "- Do not include actions for save/slack; only retrieval planning."
)


_CONTEXT_DEPENDENT_FOLLOWUP_PATTERN = re.compile(
    r"^\s*(?:\d+\s*(?:번)?|첫\s*번째|두\s*번째|세\s*번째|그거|그것|이거|저거|위에\s*것|앞에\s*것)\s*$",
    flags=re.I,
)


def _extract_message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content).strip()


def _select_planner_conversation_window(conversation: list[BaseMessage]) -> list[BaseMessage]:
    if not conversation:
        return []

    latest_human_index = -1
    for index in range(len(conversation) - 1, -1, -1):
        if isinstance(conversation[index], HumanMessage):
            latest_human_index = index
            break

    if latest_human_index < 0:
        return conversation[-1:]

    latest_human = conversation[latest_human_index]
    latest_human_text = _extract_message_text(latest_human)
    if _CONTEXT_DEPENDENT_FOLLOWUP_PATTERN.match(latest_human_text):
        start_index = max(0, latest_human_index - 2)
        return conversation[start_index : latest_human_index + 1]

    return [latest_human]


def build_planner_messages(state: GraphState, max_turns: int = 6) -> list[BaseMessage]:
    runtime = get_runtime_state(state)
    retry_context = get_retry_state(state)
    model_messages: list[BaseMessage] = [SystemMessage(content=PLANNER_SYS)]

    retry_context_message = format_retry_context_for_planner(state, retry_context)
    if retry_context_message:
        model_messages.append(SystemMessage(content=retry_context_message))

    if runtime.memory_summary:
        model_messages.extend(
            build_untrusted_memory_prompt_messages(runtime.memory_summary)
        )

    conversation = [message for message in state.get("messages", []) if not isinstance(message, ToolMessage)]
    conversation = keep_recent_messages(conversation, max_turns=max_turns)
    model_messages.extend(_select_planner_conversation_window(conversation))

    if not any(isinstance(message, HumanMessage) for message in model_messages):
        model_messages.append(HumanMessage(content=runtime.user_input.strip()))
    return model_messages
