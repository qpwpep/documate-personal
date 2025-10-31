from langgraph.graph import StateGraph, START, END

from .node import State, chatbot, add_user_message
from .tools import tavilysearch
from .edge import wire_tool_edges

def build_graph():
    # LangGraph 생성 (State 구조 기반)
    builder = StateGraph(State)
    
    # 사용자 입력을 messages에 추가하는 노드 등록
    builder.add_node("add_user_message", add_user_message)
    builder.set_entry_point("add_user_message") # START → add_user_message

    # GPT 응답 생성 노드 등록
    builder.add_node("chatbot", chatbot)
    builder.add_edge("add_user_message", "chatbot")

    # 툴 분기 조건 및 웹 검색 연동
    wire_tool_edges(builder, tavily_tool=tavilysearch)

    # 종료 설정
    builder.add_edge("chatbot", END)

    # LangGraph 앱 완성
    return builder.compile()    
