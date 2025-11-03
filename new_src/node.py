from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage

from .prompts import SYS_POLICY, needs_search, needs_save
from .llm import llm_with_tools

class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str             # 사용자의 현재 입력 (멀티턴용)
    final_answer: Optional[str] # 응답 텍스트만 따로 저장 (선택)

SAVE_HINT = "(사용자가 응답 저장을 요청했습니다. 최종 '응답 전문'을 content에 담아 'save_text' 도구를 한 번만 호출하세요.)"
SEARCH_HINT = "(이 질문은 공식/최신 문서 검색이 필요합니다. 'tavilysearch' 도구를 먼저 사용하세요.)"

def _has_hint(msgs, marker: str) -> bool:
    return any(isinstance(m, SystemMessage) and marker in m.content for m in msgs)

def chatbot(state: State):
    msgs = state["messages"]

    # Inject system policy once
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=SYS_POLICY)] + msgs

    # Look at the most recent user message
    last_user = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)

    # Heuristics: add gentle hints to steer tool decisions
    if last_user and needs_search(last_user.content) and not _has_hint(msgs, SEARCH_HINT):
        msgs.append(SystemMessage(content=SEARCH_HINT))

    if last_user and needs_save(last_user.content) and not _has_hint(msgs, SAVE_HINT):
        msgs.append(SystemMessage(content=SAVE_HINT))

    # If we just ran save_text, force the model to ACK and STOP (no more save_text)
    if msgs and isinstance(msgs[-1], ToolMessage) and getattr(msgs[-1], "name", "") == "save_text":
        msgs.append(SystemMessage(content="The content has been saved. Do NOT call save_text again in this turn. "
                                          "Acknowledge the filename returned by the tool briefly and end the turn."))

    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}

def add_user_message(state: State) -> State:
    """
    사용자의 입력(user_input)을 messages에 추가 (멀티턴 구현용)
    """
    msgs = state.get("messages", [])
    msgs.append(HumanMessage(content=state["user_input"]))
    state["messages"] = msgs
    return state
