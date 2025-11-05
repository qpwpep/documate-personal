import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment and validate required keys
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
assert os.getenv("TAVILY_API_KEY"), "Missing TAVILY_API_KEY"
# assert os.getenv("LANGSMITH_API_KEY"), "Missing LANGSMITH_API_KEY"

from .tools import tools  # relative import assumes package-style usage

# LLMs and bindings
llm = ChatOpenAI(model="gpt-4.1-mini")
llm_with_tools = llm.bind_tools(tools)

# Verbosity flag used by main loop
VERBOSE = True

# 요약 전용 llm
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
llm_summarizer = ChatOpenAI(
    model=SUMMARY_MODEL,
    temperature=0,     # 요약은 사실 중심/결정론적으로
    max_tokens=250,    # 4~5줄 목표
    timeout=60,
    max_retries=2,
    verbose=VERBOSE,   # 필요 시 내부 로그 확인
)