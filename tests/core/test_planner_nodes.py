import json
import unittest

from hypothesis import example, given, strategies as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from src.runtime.agent_runtime.llm_usage import capture_llm_usage
from src.core.contracts.boundary.debug import get_debug_state
from src.core.planner_schema import (
    PlannerOutput,
    RetrievalTask,
)
from src.core.request_contracts import RequestContract, WireRequestContract
from src.core.llm_errors import LLMCallError
from src.runtime.nodes.planner import make_planner_node

from .helpers import (
    _CapturePlannerLLM,
    _FailingPlannerLLM,
    _InvalidPlannerLLM,
    build_test_state,
)


class PlannerNodeTest(unittest.TestCase):
    pass












if __name__ == "__main__":
    unittest.main()
