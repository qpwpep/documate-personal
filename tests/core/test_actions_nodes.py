from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, PlannerState, ResponseState, RuntimeState
from src.core.contracts.graph_state import PendingAction
from src.core.request_contracts import RequestContract
from src.runtime.nodes.actions import make_action_postprocess_node
from src.infra.tools.save_text import build_save_text_tool
from src.core.slack_contract import RecipientSelector, SlackDefault
from tests.core.test_pending_action_delivery import delivery_tools


def _contract(*, save="not_requested", slack="not_requested", **updates):
    missing_info = [
        {"slot": slot, "reason": "unclear", "question": "실행 여부를 알려 주세요."}
        for slot, intent in (("save_intent", save), ("slack_intent", slack)) if intent == "unresolved"
    ]
    return RequestContract.model_validate({
        "actions": {
            "save_text": {"intent": save, "evidence_ids": ["request"]},
            "slack_notify": {"intent": slack, "evidence_ids": ["request"]},
        },
        "evidence": [{"id": "request", "turn_id": "current", "quote": "사용자 요청", "scope": "current_request", "interpretation": "instruction"}],
        "missing_info": missing_info,
        **updates,
    })


def _state(text: str, request: str = "결과를 txt로 저장해줘", *, contract=None) -> GraphState:
    contract = contract or _contract(save="requested")
    return {
        "runtime": RuntimeState(user_input=request, request_contract=contract),
        "planner": PlannerState(),
        "response": ResponseState(result=finalize_answer(text_document(text), []), kind="answer",
                                  request_id=contract.request_id, contract_revision=contract.revision),
        "messages": [HumanMessage(content=request)],
    }


























