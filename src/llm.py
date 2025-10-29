# llm.py
from typing import List
from langchain_openai import ChatOpenAI


# tools는 langchain.tools.Tool 혹은 langchain_community.tools.* 인스턴스 리스트


def build_llm_with_tools(model: str = "gpt-4.1-mini", temperature: float = 0.0, tools: List = None):
    llm = ChatOpenAI(model=model, temperature=temperature)
    if tools:
        return llm.bind_tools(tools)
    return llm