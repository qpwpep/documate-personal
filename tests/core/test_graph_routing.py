import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.contracts import GraphState, PlannerState, ResponseState
from src.contracts.boundary.graph import build_graph_state_input
from src.contracts.boundary.response import get_response_state
from src.contracts.boundary.retrieval import get_retrieval_state
from src.make_graph import build_graph
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
