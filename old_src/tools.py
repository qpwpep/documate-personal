# tools.py
import os
import json
from typing import Any, Dict, List

# 최신 권장 모듈
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool

docs_list = [
    'https://docs.python.org/3/',
    'https://git-scm.com/docs',
    'https://docs.langchain.com/oss/python/langchain/overview',
    'https://matplotlib.org/stable/users/index.html',
]

def _print_websearch_debug(query: str, results: Any, limit_chars: int = 4000) -> None:
    """항상 웹검색 로그를 표준출력으로 찍는다 (길면 자름)."""
    if os.getenv("DEBUG_WEBSEARCH", "1") not in {"1", "true", "True"}:
        return
    print(">>> [WEBSEARCH QUERY]:", query)
    try:
        pretty = json.dumps(results, ensure_ascii=False, indent=2)
    except Exception:
        pretty = str(results)
    if len(pretty) > limit_chars:
        pretty = pretty[:limit_chars] + "\n... (truncated)"
    print(">>> [WEBSEARCH RESULTS]:")
    print(pretty)


def build_websearch_tool(max_results: int = 5) -> Tool:
    """
    LangChain Tool을 반환하되 내부에서 TavilySearch를 호출하고,
    매 호출 때마다 쿼리/결과를 stdout에 출력한다.
    """
    tavily_tool = TavilySearch(max_results=max_results, include_domains=docs_list, start_date='2024-06-01')

    def _wrapped(query: str) -> Any:
        # TavilySearch는 {"query": "..."} 형태로 받습니다.
        results = tavily_tool.invoke({"query": query})
        _print_websearch_debug(query, results)
        return results

    return Tool(
        name="web_search",
        description=(
            "Tavily 웹검색을 수행하고, 쿼리와 결과를 항상 로그로 출력합니다. "
            "입력은 자연어 쿼리 문자열입니다."
        ),
        func=_wrapped,
    )