"""Selected answer subjects retain anchors without authorizing assistant guesses."""
import json

import jsonschema
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.llm_errors import LLMCallError
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.structured_schema import compile_output_schema
from src.runtime.nodes.planner import make_planner_node
from tests.core.helpers import _CapturePlannerLLM, build_test_state


CURRENT = "1번을 공식 문서에서 확인해줘"
ANSWER = "1. order 옵션\n2. copy 옵션"


def _state(current=CURRENT, answer=ANSWER, **updates):
    return build_test_state({
        "user_input": current,
        "previous_response": finalize_answer(text_document(answer), [], retrieval_required=False),
        "messages": [
            HumanMessage(content="numpy.reshape 옵션을 알려줘", id="original"),
            AIMessage(content=answer),
            HumanMessage(content=current, id="current"),
        ],
        **updates,
    })


def _proposal(*, aspects=None, quote=CURRENT, answer_quote="1. order 옵션",
              interpretation="reference", turn_id="current", scope="retrieval.reshape.aspects"):
    return {
        "use_retrieval": True,
        "tasks": [{
            "requirement_id": "reshape", "route": "docs", "query": "numpy.reshape order", "k": 3,
            "requirement": {"library": "numpy", "symbols": ["numpy.reshape"],
                            "aspects": aspects if aspects is not None else ["order"], "evidence_needs": ["parameters"]},
        }],
        "request_contract": WireRequestContract(slack_recipient={"state": "omitted"}, evidence=[{
            "id": "selection", "turn_id": turn_id, "quote": quote,
            "scope": scope, "interpretation": interpretation,
        }]).model_dump(mode="json"),
        "aspect_references": [{"requirement_id": "reshape", "aspect": "order",
                               "evidence_id": "selection", "answer_quote": answer_quote}],
    }


def _plan(proposal, state=None):
    return make_planner_node(_CapturePlannerLLM(proposal), False)(state or _state())




def test_only_declared_selected_anchor_survives_not_other_answer_terms_or_guesses():
    model = _SequencePlanner(_proposal(aspects=["order", "copy", "invented_value"]),
                             _proposal(aspects=["order", "invented_value"]))
    result = make_planner_node(model, False)(_state())
    assert result["planner"].output.tasks[0].requirement.aspects == ["order"]
    assert "unrequested_constraints_removed" in result["planner"].diagnostics.planner_warnings
    assert len(model.messages) == 2




@pytest.mark.parametrize("changes", [
    {"turn_id": "original", "quote": "numpy.reshape 옵션을 알려줘"},
    {"scope": "current_request", "interpretation": "instruction"},
    {"scope": "retrieval.other.aspects"},
    {"scope": "retrieval.reshap"},
    {"scope": "retrieval.reshape.aspects.order"},
    {"quote": "존재하지 않는 사용자 인용"},
    {"turn_id": "unknown"},
    {"answer_quote": "1. invented_value 옵션"},
    {"answer_quote": "2. copy 옵션"},
])
def test_invalid_reference_must_be_repaired_before_an_assistant_anchor_can_execute(changes):
    model = _CapturePlannerLLM(_proposal(**changes))
    with pytest.raises(LLMCallError) as caught:
        make_planner_node(model, False)(_state())
    assert caught.value.problem.code == "model_output_invalid"
    assert model.call_count == 2
















class _SequencePlanner:
    def __init__(self, *proposals):
        self.proposals = iter(proposals)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return next(self.proposals)








