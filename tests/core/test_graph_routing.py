import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.contracts import GraphState, LLMCallMetadata, PlannerState, ResponseState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.runtime.graph_builder import _instrument_stage_node
from src.runtime.make_graph import build_graph
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.session import add_user_message
from src.runtime.nodes.validation import make_pre_synthesis_validation_node, make_validate_evidence_node
from src.core.planner_schema import PlannerOutput, RetrievalTask

from .helpers import (
    _CapturePlannerLLM,
    _FailingPlannerLLM,
    _tool_payload,
)


class GraphRoutingTest(unittest.TestCase):
    pass








if __name__ == "__main__":
    unittest.main()
