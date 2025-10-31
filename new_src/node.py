from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage

from .prompts import SYS_POLICY, needs_search
from .llm import llm_with_tools

class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str             # 사용자의 현재 입력 (멀티턴용)
    final_answer: Optional[str] # 응답 텍스트만 따로 저장 (선택)

def chatbot(state: State):
    """Single-agent node that optionally injects SYS_POLICY and search hint."""
    msgs = state["messages"]
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=SYS_POLICY)] + msgs

    # Add hint if the last user message is a 'needs search' type
    last_user = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
    if last_user and needs_search(last_user.content):
        msgs.append(HumanMessage(content="(위 질문은 최신/공식 문서가 필요한 검색성 질문입니다. TavilySearch 도구를 먼저 사용해서 답변하세요.)"))

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