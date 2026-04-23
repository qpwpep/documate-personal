import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.runtime.agent_runtime.debug_collector import DebugCollector
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.actions import build_action_only_answer, make_action_postprocess_node, should_short_circuit_action_only

from .helpers import _ToolWrapper, build_legacy_state


class ActionsNodeTest(unittest.TestCase):
    pass








if __name__ == "__main__":
    unittest.main()
