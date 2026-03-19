from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ...contracts import GraphState
from ...contracts.boundary.graph import get_retry_state
from ...contracts.boundary.runtime import get_runtime_state
from ..retry import format_retry_context_for_planner
from ..session import keep_recent_messages


PLANNER_SYS = (
    "You are a retrieval planner. Return a structured plan only.\n"
    "Rules:\n"
    "- Choose retrieval routes from: docs, upload, local.\n"
    "- docs: official/latest docs on the web.\n"
    "- upload: currently uploaded-file retriever context.\n"
    "- local: local notebook/vector index examples.\n"
    "- If retrieval is unnecessary, set use_retrieval=false and tasks=[].\n"
    "- If retrieval is needed, set use_retrieval=true and include 1-3 tasks.\n"
    "- Each selected route must appear at most once.\n"
    "- Keep each task.query short and route-specific.\n"
    "- If the request is only asking to save/share/send the current answer, retrieval is unnecessary.\n"
    "- If retriever_available=true and the user is asking about the currently uploaded file, prefer upload over local.\n"
    "- Do not include actions for save/slack; only retrieval planning."
)


def build_planner_messages(state: GraphState, max_turns: int = 6) -> list[BaseMessage]:
    runtime = get_runtime_state(state)
    retry_context = get_retry_state(state)
    model_messages: list[BaseMessage] = [SystemMessage(content=PLANNER_SYS)]
    model_messages.append(
        SystemMessage(content=f"[Planner Context]\nretriever_available={bool(runtime.retriever)}")
    )

    retry_context_message = format_retry_context_for_planner(state, retry_context)
    if retry_context_message:
        model_messages.append(SystemMessage(content=retry_context_message))

    if runtime.memory_summary:
        model_messages.append(SystemMessage(content=f"[Conversation Summary]\n{runtime.memory_summary}"))

    conversation = [message for message in state.get("messages", []) if not isinstance(message, ToolMessage)]
    conversation = keep_recent_messages(conversation, max_turns=max_turns)
    latest_conversation: list[BaseMessage] = []
    latest_human_index = -1
    for index in range(len(conversation) - 1, -1, -1):
        if isinstance(conversation[index], HumanMessage):
            latest_human_index = index
            break
    if latest_human_index >= 0:
        latest_conversation.append(conversation[latest_human_index])
    else:
        latest_conversation = conversation[-1:]
    model_messages.extend(latest_conversation)

    if not any(isinstance(message, HumanMessage) for message in model_messages):
        model_messages.append(HumanMessage(content=runtime.user_input.strip()))
    return model_messages
