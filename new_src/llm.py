import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from .tools import tools  # relative import assumes package-style usage

# Load environment and validate required keys
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
assert os.getenv("TAVILY_API_KEY"), "Missing TAVILY_API_KEY"
assert os.getenv("LANGSMITH_API_KEY"), "Missing LANGSMITH_API_KEY"

# LLMs and bindings
llm = ChatOpenAI(model="gpt-4.1-mini")
llm_with_tools = llm.bind_tools(tools)

# Verbosity flag used by main loop
VERBOSE = False