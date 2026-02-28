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


SAVE_HINT = "(사용자가 응답 저장을 요청했습니다. 최종 '응답 전문'을 content에 담아 'save_text' 도구를 한 번만 호출하세요.)"
SEARCH_HINT = "(이 질문은 공식/최신 문서 검색이 필요합니다. 'tavily_search' 도구를 먼저 사용하세요.)"
RAG_HINT = "(이 요청은 로컬 노트북/예제 기반 지식 검색이 필요합니다. 'rag_search' 도구를 사용하세요.)"
SLACK_HINT = (
    "(사용자가 Slack 전송을 요청했습니다. 최종 답변을 'slack_notify' 도구로 보내세요. "
    "가능하면 channel_id 또는 user_id/email 인자를 채워주세요.)"
)


def _has_hint(msgs: list[AnyMessage], marker: str) -> bool:
    return any(isinstance(m, SystemMessage) and marker in m.content for m in msgs)


def _inject_uploaded_context_if_any(
    state: State,
    msgs: list[AnyMessage],
    verbose: bool = False,
) -> list[AnyMessage]:
    retriever = state.get("retriever")
    if not retriever:
        return msgs

    last_user = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
    if not last_user or not last_user.content.strip():
        return msgs

    try:
        docs = retriever.invoke(last_user.content)
        if not docs:
            return msgs
        lines = []
        for doc in docs[:4]:
            src = doc.metadata.get("source", "uploaded")
            snippet = (doc.page_content or "").strip().replace("\n", " ")
            if len(snippet) > 500:
                snippet = snippet[:500] + " ..."
            lines.append(f"- {snippet}\n  [◆ 업로드 파일] {src}")
        context_block = (
            "아래는 사용자가 업로드한 파일에서 검색된 관련 구문입니다. 가능한 한 이를 우선 참고해 답변하세요:\n"
            + "\n".join(lines)
        )
        return msgs + [SystemMessage(content=context_block)]
    except Exception as exc:
        if verbose:
            print(f"[retriever] context injection failed: {exc}")
        return msgs


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
    "너는 회의록 비서다. 아래 대화의 핵심을 **한국어 4~5줄**로 요약하라.\n"
    "- 주제/결론/결정/중요한 코드/버전/URL만 유지\n"
    "- 중복/군더더기 제거, 불확실하면 명시\n"
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
                SystemMessage(content=f"[이전 요약]\n{state['memory_summary']}"),
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

        if model_msgs and isinstance(model_msgs[-1], ToolMessage) and getattr(model_msgs[-1], "name", "") == "save_text":
            model_msgs.append(
                SystemMessage(
                    content=(
                        "The content has been saved. Do NOT call save_text again in this turn. "
                        "Acknowledge the filename returned by the tool briefly and end the turn."
                    )
                )
            )

        model_msgs = _inject_uploaded_context_if_any(state, model_msgs, verbose=verbose)
        response: AIMessage = llm_with_tools.invoke(model_msgs)
        return {"messages": [response]}

    return chatbot
