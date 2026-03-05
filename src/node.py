from typing import Annotated, Any, List, Optional

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import add_messages
from typing_extensions import TypedDict

from .prompts import SYS_POLICY, needs_rag, needs_save, needs_search, needs_slack


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str
    final_answer: Optional[str]
    retriever: Optional[Any]
    memory_summary: Optional[str]


SAVE_HINT = (
    "(The user asked to save the answer. Call 'save_text' exactly once with the final "
    "answer text as content.)"
)
SEARCH_HINT = "(This question needs official or latest docs. Use 'tavily_search' first.)"
RAG_HINT = "(This question needs local notebook examples. Use 'rag_search'.)"
UPLOAD_HINT = "(An uploaded-file retriever is available. Use 'upload_search' for uploaded-file evidence.)"
SLACK_HINT = (
    "(The user asked to send to Slack. Use 'slack_notify' with channel_id or user_id/email when available.)"
)


def _has_hint(msgs: list[AnyMessage], marker: str) -> bool:
    return any(isinstance(m, SystemMessage) and marker in m.content for m in msgs)


def add_user_message(state: State) -> State:
    msgs = list(state.get("messages", []))
    msgs.append(HumanMessage(content=state["user_input"]))
    state["messages"] = msgs
    return state


def _keep_recent_messages(messages: List[BaseMessage], max_turns: int = 6) -> List[BaseMessage]:
    if not messages:
        return messages
    window_size = max_turns * 2 + 2
    return messages[-window_size:]


_SUMMARY_SYS = (
    "Summarize the older conversation in 4-5 lines.\n"
    "- Keep topic, conclusions, decisions, key code/version/URL.\n"
    "- Remove duplication.\n"
    "- If uncertain, state uncertainty explicitly.\n"
)


def make_summarize_node(llm_summarizer: Any, verbose: bool, max_turns: int = 6):
    def summarize_old_messages(state: State) -> State:
        msgs: List[BaseMessage] = state.get("messages", [])
        recent_window = _keep_recent_messages(msgs, max_turns=max_turns)
        if len(recent_window) == len(msgs):
            return state

        cutoff = len(msgs) - len(recent_window)
        old = msgs[:cutoff]
        recent = msgs[cutoff:]

        try:
            summary = llm_summarizer.invoke([SystemMessage(content=_SUMMARY_SYS)] + old).content.strip()
        except Exception as exc:
            if verbose:
                print(f"[summary] failed: {exc}")
            state["messages"] = recent
            return state

        prev = (state.get("memory_summary") or "").strip()
        merged = (prev + ("\n" if prev else "") + summary).strip()
        state["memory_summary"] = merged
        state["messages"] = recent
        if verbose:
            print(f"[summary] merged ({cutoff} msgs -> 4~5 lines)")
        return state

    return summarize_old_messages


def make_chatbot_node(llm_with_tools: Any, verbose: bool, max_turns: int = 6):
    def chatbot(state: State):
        msgs = state.get("messages", [])
        model_msgs: List[BaseMessage] = list(msgs)

        if not model_msgs or not isinstance(model_msgs[0], SystemMessage):
            model_msgs = [SystemMessage(content=SYS_POLICY)] + model_msgs

        if state.get("memory_summary"):
            model_msgs = [
                model_msgs[0],
                SystemMessage(content=f"[Conversation Summary]\n{state['memory_summary']}"),
            ] + model_msgs[1:]

        before = len(model_msgs)
        model_msgs = _keep_recent_messages(model_msgs, max_turns=max_turns)
        after = len(model_msgs)
        if verbose and before != after:
            print(f"[trim] messages for model input: {before} -> {after}")

        last_user = next((m for m in reversed(model_msgs) if isinstance(m, HumanMessage)), None)
        if last_user:
            content = last_user.content
            if needs_search(content) and not _has_hint(model_msgs, SEARCH_HINT):
                model_msgs.append(SystemMessage(content=SEARCH_HINT))
            if needs_rag(content) and not _has_hint(model_msgs, RAG_HINT):
                model_msgs.append(SystemMessage(content=RAG_HINT))
            if needs_save(content) and not _has_hint(model_msgs, SAVE_HINT):
                model_msgs.append(SystemMessage(content=SAVE_HINT))
            if needs_slack(content) and not _has_hint(model_msgs, SLACK_HINT):
                model_msgs.append(SystemMessage(content=SLACK_HINT))

        if state.get("retriever") and not _has_hint(model_msgs, UPLOAD_HINT):
            model_msgs.append(SystemMessage(content=UPLOAD_HINT))

        if (
            model_msgs
            and isinstance(model_msgs[-1], ToolMessage)
            and getattr(model_msgs[-1], "name", "") == "save_text"
        ):
            model_msgs.append(
                SystemMessage(
                    content=(
                        "The content has been saved. Do NOT call save_text again in this turn. "
                        "Acknowledge the filename returned by the tool briefly and end the turn."
                    )
                )
            )

        response: AIMessage = llm_with_tools.invoke(model_msgs)
        return {"messages": [response]}

    return chatbot
