from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from .node import State, chatbot, add_user_message, summarize_old_messages
from .tools import tavilysearch, rag_search_tool, save_text_tool
from .edge import wire_tool_edges


def build_graph():
    # LangGraph 생성 (State 구조 기반)
    builder = StateGraph(State)

    # 사용자 입력을 messages에 추가하는 노드 등록
    builder.add_node("add_user_message", add_user_message)
    builder.set_entry_point("add_user_message")  # START → add_user_message

    # 오래된 메시지를 4~5줄로 요약하고 state를 '최근 6턴'으로 정리
    builder.add_node("summarize_old_messages", summarize_old_messages)
    
    # GPT 응답 생성 노드 등록
    builder.add_node("chatbot", chatbot)
    
    # 흐름: add_user_message → summarize_old_messages → chatbot
    builder.add_edge("add_user_message", "summarize_old_messages")
    builder.add_edge("summarize_old_messages", "chatbot")

    # ✅ 세 개의 툴(TavilySearch, RAGSearch, SaveText)을 단일 ToolNode에 연결
    tool_node = ToolNode(tools=[tavilysearch, rag_search_tool, save_text_tool])
    builder.add_node("tools", tool_node)

    # 모델이 툴 호출이 필요하면 tools로, 아니면 종료
    builder.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})

    # 툴 실행 후에는 다시 chatbot으로
    builder.add_edge("tools", "chatbot")

    # LangGraph 앱 완성
    return builder.compile()
