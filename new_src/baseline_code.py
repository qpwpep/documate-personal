import re
import os
import json

from dotenv import load_dotenv
from typing import Any, List, Optional, TypedDict, Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
assert os.getenv("TAVILY_API_KEY"), "Missing TAVILY_API_KEY"
assert os.getenv("LANGSMITH_API_KEY"), "Missing LANGSMITH_API_KEY"

SYS_POLICY = """You are a grounded assistant.
- For queries asking for latest/official docs or anything time-sensitive, you MUST call TavilySearch first, then answer with concise bullets and cite sources.
"""

NEED_SEARCH_PATTERNS = [
    r"\b(latest|official|docs?|documentation|reference)\b",
    r"(최신|공식|문서|레퍼런스|가격|발매|지원 버전|변경점|로드맵)"
]
def needs_search(text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in NEED_SEARCH_PATTERNS)

VERBOSE = False

tavilysearch = TavilySearch(max_results=3)
tools = [tavilysearch]
llm = ChatOpenAI(model="gpt-4.1-mini")
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

graph_builder = StateGraph(State)


def chatbot(state: State): # Agent
    # 항상 시스템 정책을 맨 앞에 붙여준다
    msgs = state["messages"]
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=SYS_POLICY)] + msgs

    # 사용자가 검색성 질문이면 힌트 메시지 추가
    last_user = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
    if last_user and needs_search(last_user.content):
        msgs.append(HumanMessage(content="(위 질문은 최신/공식 문서가 필요한 검색성 질문입니다. TavilySearch 도구를 먼저 사용해서 답변하세요.)"))

    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tavilysearch]) # 웹검색도구를 실행하는 노드
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition, # 도구 호출이 발생했는지 발생하지 않았는 지 판단해주는 함수
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Saved graph to graph.png")
except Exception as e:
    # graphviz/mermaid 설치가 없으면 실패할 수 있음
    pass


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        invoke_input = {
            "messages": [HumanMessage(content=user_input)]
        }
        response = graph.invoke(invoke_input)

        if VERBOSE:
            for msg in response["messages"]:
                msg.pretty_print()
        else:
            # 마지막 AI 메시지 출력
            for m in reversed(response["messages"]):
                if isinstance(m, AIMessage):
                    print(m.content)
                    break
    except Exception as e:
        print("Error:", e)
        break