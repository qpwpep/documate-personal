import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.make_graph import build_graph
from src.contracts.graph_state import GraphState, PlannerState, ResponseState, build_graph_state_input, response_state, retrieval_state
from src.nodes.planner import make_planner_node
from src.nodes.retrieval import make_retrieve_dispatch_node
from src.nodes.session import add_user_message
from src.nodes.validation import make_validate_evidence_node
from src.planner_schema import PlannerOutput

from .helpers import (
    _CapturePlannerLLM,
    _FailingPlannerLLM,
    _ToolWrapper,
    _tool_payload,
)


class GraphRoutingTest(unittest.TestCase):
    pass






if __name__ == "__main__":
    unittest.main()
