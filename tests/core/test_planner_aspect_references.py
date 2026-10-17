"""Selected answer subjects retain anchors without authorizing assistant guesses."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import finalize_answer, text_document
from src.core.llm_errors import LLMCallError
from src.core.request_contracts import WireRequestContract
from src.runtime.nodes.planner import make_planner_node
from tests.core.helpers import _CapturePlannerLLM, build_test_state


CURRENT = "1번을 공식 문서에서 확인해줘"
ANSWER = "1. order 옵션\n2. copy 옵션"


def _state():
    return build_test_state({
        "user_input": CURRENT,
        "previous_response": finalize_answer(text_document(ANSWER), [], retrieval_required=False),
        "messages": [
            HumanMessage(content="numpy.reshape 옵션을 알려줘", id="original"),
            AIMessage(content=ANSWER),
            HumanMessage(content=CURRENT, id="current"),
        ],
    })


def _proposal(*, aspects=None, answer_quote="1. order 옵션"):
    return {
        "use_retrieval": True,
        "tasks": [{
            "requirement_id": "reshape", "route": "docs", "query": "numpy.reshape order", "k": 3,
            "requirement": {"library": "numpy", "symbols": ["numpy.reshape"],
                            "aspects": aspects if aspects is not None else ["order"], "evidence_needs": ["parameters"]},
        }],
        "request_contract": WireRequestContract(slack_recipient={"state": "omitted"}, evidence=[{
            "id": "selection", "turn_id": "current", "quote": CURRENT,
            "scope": "retrieval.reshape.aspects", "interpretation": "reference",
        }]).model_dump(mode="json"),
        "aspect_references": [{"requirement_id": "reshape", "aspect": "order",
                               "evidence_id": "selection", "answer_quote": answer_quote}],
    }


def test_only_declared_selected_anchor_survives_not_other_answer_terms_or_guesses():
    model = _SequencePlanner(_proposal(aspects=["order", "copy", "invented_value"]),
                             _proposal(aspects=["order", "invented_value"]))
    result = make_planner_node(model, False)(_state())
    assert result["planner"].output.tasks[0].requirement.aspects == ["order"]
    assert "unrequested_constraints_removed" in result["planner"].diagnostics.planner_warnings
    assert len(model.messages) == 2


def test_invalid_reference_must_be_repaired_before_an_assistant_anchor_can_execute():
    model = _CapturePlannerLLM(_proposal(answer_quote="2. copy 옵션"))
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
