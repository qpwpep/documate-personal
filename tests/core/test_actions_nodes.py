from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, PlannerState, ResponseState, RuntimeState
from src.runtime.nodes.actions import make_action_postprocess_node, should_short_circuit_action_only


def _state(text: str, request: str = "결과를 txt로 저장해줘") -> GraphState:
    return {
        "runtime": RuntimeState(user_input=request),
        "planner": PlannerState(),
        "response": ResponseState(result=finalize_answer(text_document(text), [])),
        "messages": [HumanMessage(content=request)],
    }














