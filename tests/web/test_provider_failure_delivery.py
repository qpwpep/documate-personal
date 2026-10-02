"""Real SDK, graph, service, HTTP/SSE, client and UI share the failure contract."""
from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI
from streamlit.testing.v1 import AppTest

from src.app.client import AgentRequestContext, AgentSessionClient


