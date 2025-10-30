# make_graph.py
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# 로컬/상대 임포트 환경에 맞춰 수정해 주세요
try:
    # 패키지 형태
    from .node import GraphState, route, call_model, collect_observation, finalize_answer
    from .llm import build_llm_with_tools
    from .tools import build_websearch_tool
    from .prompts import SYSTEM_FINALIZE
    from .edge import need_tools # 선택 파일
except Exception:
    # 단일 폴더 스크립트 실행용 fallback
    from node import GraphState, route, call_model, collect_observation, finalize_answer
    from llm import build_llm_with_tools
    from tools import build_websearch_tool
    from prompts import SYSTEM_FINALIZE
try:
    from edge import need_tools
except Exception:
    need_tools = None




def build_graph(model: str = "gpt-4.1-mini", max_results: int = 5):
    # Tools
    web_tool = build_websearch_tool(max_results=max_results)
    tools = [web_tool]


    # LLM with tool-calling
    llm_with_tools = build_llm_with_tools(model=model, tools=tools)


    g = StateGraph(GraphState)


    # Nodes
    g.add_node("router", route)
    g.add_node("model", lambda s: call_model(s, llm_with_tools))
    g.add_node("tools", ToolNode(tools=tools))
    g.add_node("collect", collect_observation)
    g.add_node("finalize", lambda s: finalize_answer(s, system_prompt=SYSTEM_FINALIZE, model=model))


    # Edges
    g.set_entry_point("router")
    g.add_edge("router", "model")


    # conditional edge: model → tools or finalize
    if need_tools:
        g.add_conditional_edges("model", need_tools, {"tools": "tools", "finalize": "finalize"})
    else:
        # 내장 조건자 (간단 버전)
        def _need_tools(state: GraphState):
            last = state["messages"][-1]
            return "tools" if getattr(last, "tool_calls", None) else "finalize"
        g.add_conditional_edges("model", _need_tools, {"tools": "tools", "finalize": "finalize"})


    g.add_edge("tools", "collect")
    g.add_edge("collect", "model")
    g.add_edge("finalize", END)


    return g.compile()