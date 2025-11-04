from typing import Annotated, Any, Optional, List
from typing_extensions import TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage

from .prompts import SYS_POLICY, needs_search, needs_save
from .llm import llm_with_tools, VERBOSE

class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str             # 사용자의 현재 입력 (멀티턴용)
    final_answer: Optional[str] # 응답 텍스트만 따로 저장 (선택)

SAVE_HINT = "(사용자가 응답 저장을 요청했습니다. 최종 '응답 전문'을 content에 담아 'save_text' 도구를 한 번만 호출하세요.)"
SEARCH_HINT = "(이 질문은 공식/최신 문서 검색이 필요합니다. 'tavilysearch' 도구를 먼저 사용하세요.)"

def _has_hint(msgs, marker: str) -> bool:
    return any(isinstance(m, SystemMessage) and marker in m.content for m in msgs)

def add_user_message(state: State) -> State:
    """
    사용자의 입력(user_input)을 messages에 추가 (멀티턴 구현용)
    """
    msgs = state.get("messages", [])
    msgs.append(HumanMessage(content=state["user_input"]))
    state["messages"] = msgs
    return state

def _keep_recent_messages(messages: List[BaseMessage], max_turns: int = 6) -> List[BaseMessage]:
    """
    Human+AI 한 쌍을 '1턴'으로 보고, 모델 입력용으로 최근 max_turns 턴만 유지
    - state['messages'] 자체는 건드리지 않습니다. (전체 히스토리는 유지)
    - 모델 호출 직전에 **입력 복사본**에만 트리밍을 적용합니다.
    - 간단히 길이 기준(2*턴 + 2)로 자릅니다. (System 1 + 여유분 1 가정)
    - 필요하면 턴 페어링/토큰 카운트 기반으로 정교화할 수 있습니다.
    """
    if not messages:
        return messages
    window_size = max_turns * 2 + 2  # System 1개 + (Human/AI) * N + 여유
    return messages[-window_size:]

def chatbot(state: State):
    msgs = state["messages"]
    
    # 모델 입력용 복사본 생성
    model_msgs: List[BaseMessage] = list(msgs)

    # Inject system policy once (입력 복사본에만)
    if not model_msgs or not isinstance(model_msgs[0], SystemMessage):
        model_msgs = [SystemMessage(content=SYS_POLICY)] + model_msgs

    # 모델 입력에 한해 최근 N턴만 반영
    before = len(model_msgs)
    model_msgs = _keep_recent_messages(model_msgs, max_turns=6)
    after = len(model_msgs)
    if VERBOSE and before != after:
        print(f"[trim] messages for model input: {before} -> {after}")

    # Look at the most recent user message (트리밍 이후 기준으로 판단)
    last_user = next((m for m in reversed(model_msgs) if isinstance(m, HumanMessage)), None)

    # Heuristics: add gentle hints to steer tool decisions
    if last_user and needs_search(last_user.content) and not _has_hint(model_msgs, SEARCH_HINT):
        model_msgs.append(SystemMessage(content=SEARCH_HINT))

    if last_user and needs_save(last_user.content) and not _has_hint(model_msgs, SAVE_HINT):
        model_msgs.append(SystemMessage(content=SAVE_HINT))

    # If we just ran save_text, force the model to ACK and STOP (no more save_text)
    if model_msgs and isinstance(model_msgs[-1], ToolMessage) and getattr(model_msgs[-1], "name", "") == "save_text":
        model_msgs.append(SystemMessage(
            content=(
                "The content has been saved. Do NOT call save_text again in this turn. "
                "Acknowledge the filename returned by the tool briefly and end the turn."
            )
        ))

    # invoke에는 잘라낸 입력 복사본(model_msgs)을 사용, 원본 msgs는 그대로 보존
    response: AIMessage = llm_with_tools.invoke(model_msgs)
    return {"messages": [response]}

