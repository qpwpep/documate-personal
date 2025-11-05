from typing import Annotated, Any, Optional, List
from typing_extensions import TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage

from .prompts import SYS_POLICY, needs_search, needs_save, needs_rag
from .llm import llm_with_tools, VERBOSE, llm_summarizer

class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # LangGraph가 자동 누적
    user_input: str                                      # 현재 사용자 입력
    final_answer: Optional[str]                          # (선택) 응답 텍스트
    retriever: Optional[Any]                             # (선택) 세션별 벡터검색기
    memory_summary: Optional[str]                        # 이전 대화 요약(4~5줄)


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

"""
오래된 대화를 4~5줄로 요약해 memory_summary에 누적 + state를 최근 N턴으로 정리
* 팀 로직을 건드리지 않기 위해 node 내부에서만 수행
* make_graph에서 add_user_message 다음에 이 노드를 한번 거치게 연결
"""
_SUMMARY_SYS = (
    "너는 회의록 비서다. 아래 대화의 핵심을 **한국어 4~5줄**로 요약하라.\n"
    "- 주제/결론/결정/중요한 코드/버전/URL만 유지\n"
    "- 중복/군더더기 제거, 불확실하면 명시\n"
)

def summarize_old_messages(state: State, max_turns: int = 6) -> State:
    """
    전체 messages 길이가 '최근 N턴 윈도우'보다 길어졌을 때만:
      - 오래된 구간만 요약하여 memory_summary에 누적
      - state['messages']는 최근 N턴만 남김 (상태 자체를 가볍게 유지)
    """
    msgs: List[BaseMessage] = state.get("messages", [])
    recent_window = _keep_recent_messages(msgs, max_turns=max_turns)
    if len(recent_window) == len(msgs):
        # 아직 짧으면 요약 불필요
        return state

    # 오래된 구간(old)과 최근 구간(recent) 분리
    cutoff = len(msgs) - len(recent_window)
    old = msgs[:cutoff]
    recent = msgs[cutoff:]

    # 오래된 구간만 요약 시도
    try:
        summary = llm_summarizer.invoke(
            [SystemMessage(content=_SUMMARY_SYS)] + old
        ).content.strip()
    except Exception as e:
        if VERBOSE:
            print(f"[summary] failed: {e}")
        # 요약 실패해도 최근만 유지(서비스 지속)
        state["messages"] = recent
        return state

    # 이전 요약이 있으면 이어붙임(누적)
    prev = (state.get("memory_summary") or "").strip()
    merged = (prev + ("\n" if prev else "") + summary).strip()

    state["memory_summary"] = merged          # 4~5줄 요약 누적
    state["messages"] = recent                # 상태는 최근 N턴만 보존 (메모리/비용 절감)
    if VERBOSE:
        print(f"[summary] merged ({cutoff} msgs -> 4~5 lines)")

    return state

def chatbot(state: State):
    # 방어적 시작: messages가 없을 수도 있으므로 get 사용
    msgs = state.get("messages", [])
    
    # 모델 입력용 복사본 생성
    model_msgs: List[BaseMessage] = list(msgs)

    # 정책(System) 1회 주입 — 입력 복사본에만
    if not model_msgs or not isinstance(model_msgs[0], SystemMessage):
        model_msgs = [SystemMessage(content=SYS_POLICY)] + model_msgs
        
    # 이전 요약이 있으면, 정책 바로 뒤에 '[이전 요약]' SystemMessage로 짧게 주입
    if state.get("memory_summary"):
        model_msgs = [model_msgs[0], SystemMessage(content=f"[이전 요약]\n{state['memory_summary']}")] + model_msgs[1:]

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
