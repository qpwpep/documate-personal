# node.py
from typing import List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.graph import add_messages

class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    gathered_urls: List[str]
    used_web: bool


# --- Router ---------------------------------------------------------------


def route(state: GraphState):
    msgs = state.get("messages") or []
    if not msgs:
        msgs = [HumanMessage(state["query"])]
    # state 업데이트
    return {"messages": msgs}


# --- Model ----------------------------------------------------------------


def call_model(state: GraphState, llm_with_tools):
    msgs = state["messages"]
    # 안전장치: 혹시라도 첫 메시지가 tool이면 앞에 Human을 보강
    if msgs and getattr(msgs[0], "type", "") == "tool":
        msgs = [HumanMessage(state["query"])] + msgs

    resp = llm_with_tools.invoke(msgs)  # AIMessage (tool_calls 포함 가능)

    # 누적 append
    return {"messages": msgs + [resp]}


# --- Collect (observation 정리) -------------------------------------------


def collect_observation(state: GraphState):
    urls: List[str] = state.get("gathered_urls", [])
    last = state["messages"][-1]

    # ToolMessage의 텍스트에서 URL 추출(간단 regex)
    import re
    if isinstance(last, ToolMessage) and isinstance(last.content, str):
        urls += re.findall(r"https?://\S+", last.content)

    # 중복 제거, 순서 유지
    urls = list(dict.fromkeys(urls))
    used_web = bool(state.get("used_web")) or isinstance(last, ToolMessage)

    return {"gathered_urls": urls, "used_web": used_web}


# --- Finalize -------------------------------------------------------------


def finalize_answer(state: GraphState, system_prompt: str, model: str = "gpt-4.1-mini"):
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = [
        SystemMessage(content=system_prompt),
        *state["messages"],  
        HumanMessage(content="위 대화와 도구 관찰을 바탕으로 최종 답변만 출력하세요."),
    ]
    final = llm.invoke(prompt)
    return {"messages": state["messages"] + [final]}