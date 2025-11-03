from typing import Annotated, Optional, Any
from typing_extensions import TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage

from .prompts import SYS_POLICY, needs_search, needs_rag, needs_save
from .llm import llm_with_tools


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str
    final_answer: Optional[str]
    retriever: Optional[Any]


SAVE_HINT = "(사용자가 응답 저장을 요청했습니다. 최종 '응답 전문'을 content에 담아 'save_text' 도구를 한 번만 호출하세요.)"
SEARCH_HINT = "(이 질문은 공식/최신 문서 검색이 필요합니다. 'tavilysearch' 도구를 먼저 사용하세요.)"
RAG_HINT = "(이 요청은 로컬 노트북/예제 기반 지식 검색이 필요합니다. 'rag_search' 도구를 사용하세요.)"


def _has_hint(msgs, marker: str) -> bool:
    return any(isinstance(m, SystemMessage) and marker in m.content for m in msgs)

def _inject_uploaded_context_if_any(state: State, msgs: list[AnyMessage]) -> list[AnyMessage]:
    """If a session retriever exists, fetch short snippets for the last user query and inject as context."""
    retriever = state.get("retriever")
    if not retriever:
        return msgs

    last_user = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
    if not last_user or not last_user.content.strip():
        return msgs

    try:
        docs = retriever.get_relevant_documents(last_user.content)
        if not docs:
            return msgs
        lines = []
        for d in docs[:4]:
            src = d.metadata.get("source", "uploaded")
            snippet = (d.page_content or "").strip().replace("\n", " ")
            if len(snippet) > 500:
                snippet = snippet[:500] + " …"
            lines.append(f"- {snippet}\n  [◆ 업로드 파일] {src}")
        context_block = "아래는 사용자가 업로드한 파일에서 검색된 관련 구문입니다. 가능한 한 이를 우선 참고해 답변하세요:\n" + "\n".join(lines)
        msgs = msgs + [SystemMessage(content=context_block)]
    except Exception:
        pass
    return msgs

def chatbot(state: State):
    # ✅ Be defensive
    msgs = state.get("messages", [])

    # Inject SYS policy once
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=SYS_POLICY)] + msgs

    # Heuristic nudges
    last_user = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
    if last_user:
        content = last_user.content
        if needs_search(content) and not _has_hint(msgs, SEARCH_HINT):
            msgs.append(SystemMessage(content=SEARCH_HINT))
        if needs_rag(content) and not _has_hint(msgs, RAG_HINT):
            msgs.append(SystemMessage(content=RAG_HINT))
        if needs_save(content) and not _has_hint(msgs, SAVE_HINT):
            msgs.append(SystemMessage(content=SAVE_HINT))

    # If we just ran save_text, ACK and stop further save calls this turn
    if msgs and isinstance(msgs[-1], ToolMessage) and getattr(msgs[-1], "name", "") == "save_text":
        msgs.append(SystemMessage(
            content="The content has been saved. Do NOT call save_text again in this turn. "
                    "Acknowledge the filename returned by the tool briefly and end the turn."
        ))

    msgs = _inject_uploaded_context_if_any(state, msgs)

    # Call the tool-enabled LLM
    ai = llm_with_tools.invoke(msgs)

    # ✅ Preserve full history (append the AI message)
    return {"messages": msgs + [ai]}

def add_user_message(state: State) -> State:
    msgs = state.get("messages", [])
    msgs.append(HumanMessage(content=state["user_input"]))
    state["messages"] = msgs
    return state
