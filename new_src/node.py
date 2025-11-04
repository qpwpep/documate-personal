from typing import Annotated, Any, Optional, List
from typing_extensions import TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage

from .prompts import SYS_POLICY, needs_search, needs_save, needs_rag
from .llm import llm_with_tools, VERBOSE

# # FastAPI 실행 상태에서 로그 확인을 위해 추가
# import logging
# logger = logging.getLogger("uvicorn")

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
        # docs = retriever.get_relevant_documents(last_user.content) 
        docs = retriever.invoke(last_user.content) # 최신 버전에서는 invoke 사용
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
    except Exception as e:
        # logger.info(f"[test] exception: {e}")
        print(f"[test] exception: {e}")
        pass
    return msgs

def add_user_message(state: State) -> State:
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
    # ✅ Be defensive
    msgs = state.get("messages", [])
    
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
    if last_user:
        content = last_user.content
        if needs_search(content) and not _has_hint(model_msgs, SEARCH_HINT):
            model_msgs.append(SystemMessage(content=SEARCH_HINT))
        if needs_rag(content) and not _has_hint(model_msgs, RAG_HINT):
            model_msgs.append(SystemMessage(content=RAG_HINT))
        if needs_save(content) and not _has_hint(model_msgs, SAVE_HINT):
            model_msgs.append(SystemMessage(content=SAVE_HINT))

    # If we just ran save_text, force the model to ACK and STOP (no more save_text)
    if model_msgs and isinstance(model_msgs[-1], ToolMessage) and getattr(model_msgs[-1], "name", "") == "save_text":
        model_msgs.append(SystemMessage(
            content=(
                "The content has been saved. Do NOT call save_text again in this turn. "
                "Acknowledge the filename returned by the tool briefly and end the turn."
            )
        ))

    model_msgs = _inject_uploaded_context_if_any(state, model_msgs)

    # invoke에는 잘라낸 입력 복사본(model_msgs)을 사용, 원본 msgs는 그대로 보존
    response: AIMessage = llm_with_tools.invoke(model_msgs)
    return {"messages": [response]}

    # upload branch
    # # Call the tool-enabled LLM
    # ai = llm_with_tools.invoke(msgs)

    # # ✅ Preserve full history (append the AI message)
    # return {"messages": msgs + [ai]}
