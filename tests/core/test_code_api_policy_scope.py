"""API identity criteria follow typed content roles at the public post boundary."""
from __future__ import annotations

import json

import pytest

from src.core.answer_schema import AnswerDocument, finalize_answer
from src.core.contracts import GraphState
from src.core.prompts import CODE_API_IDENTITY_POLICY
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core import (
    test_decisions_validation,
    test_inference_api_identity_live,
    test_post_source_quote_live,
    test_transform_answer_delivery_live,
)
from tests.core.test_decisions_validation import DecisionsBoundary


def _state(case: str) -> GraphState:
    if case == "documentation_prose":
        return test_decisions_validation._state()
    if case == "interaction_transform":
        return test_transform_answer_delivery_live._post_state(
            test_transform_answer_delivery_live._frozen_state("frozen"),
            test_transform_answer_delivery_live.CORRECT_TEXT,
        )
    if case in {"source_api_prose", "inference_api_prose"}:
        state = test_inference_api_identity_live._state("actual_bad")
        if case == "source_api_prose":
            document = state["response"].result.content.model_copy(deep=True)
            document.blocks[1].content[0].basis = "source"
            state["response"] = state["response"].model_copy(update={
                "result": finalize_answer(document, state["response"].evidence_packet, retrieval_required=True),
            })
        return state
    if case == "altered_code_quote":
        return test_post_source_quote_live._state("fabricated-source-quote")
    state = test_post_source_quote_live._state("exact-source-and-location")
    if case == "excerpt":
        evidence, = state["response"].evidence_packet
        document = AnswerDocument.model_validate({"blocks": [{
            "type": "paragraph", "content": [{
                "text": test_post_source_quote_live.SOURCE, "basis": "excerpt", "refs": [evidence.id],
            }],
        }]})
        state["response"] = state["response"].model_copy(update={
            "result": finalize_answer(document, [evidence], retrieval_required=True),
        })
    return state


@pytest.mark.parametrize("case,question_name,expected_policy", [
    ("source_location", "grounded_1", False),
    ("source_code_quote", "grounded_0", False),
    ("altered_code_quote", "grounded_0", False),
    ("excerpt", "grounded_0", False),
    ("interaction_transform", "grounded_0", False),
    ("documentation_prose", "grounded_0", False),
    ("source_api_prose", "grounded_2", True),
    ("inference_api_prose", "grounded_2", True),
])
def test_post_api_identity_policy_respects_the_typed_content_role(
    case: str, question_name: str, expected_policy: bool,
) -> None:
    state = _state(case)
    boundary = DecisionsBoundary()
    make_post_synthesis_validation_node(False, decisions_client=boundary, pass_threshold=0.8)(state)
    call, = [call for call in boundary.requests
             if any(question["name"] == question_name for question in call["questions"])]
    question, = [question for question in call["questions"] if question["name"] == question_name]
    payload = json.loads(call["input"])
    unit, = [unit for unit in payload["units"] if unit["name"] == question_name]

    if case == "source_location":
        assert unit["source_location"] == {"ref": "e1", "start": 59, "end": 89}
        assert unit["text"] == "branch.py:6"
    elif case in {"source_code_quote", "altered_code_quote", "excerpt"}:
        assert unit["block_type"] == ("paragraph" if case == "excerpt" else "code")
        assert unit["basis"] == ("excerpt" if case == "excerpt" else "source")
        assert "support every material claim" in question["instructions"]
    elif case == "interaction_transform":
        assert unit["basis"] == "interaction"
        assert payload["confirmed_request"]["body"]["kind"] == "transform_answer"
        assert payload["bound_source_response"]["content_hash"] == state["runtime"].previous_response.content_hash
    elif case == "documentation_prose":
        assert unit["basis"] == "source"
        assert unit["block_type"] == "paragraph"
        assert unit["cited_sources"][0]["kind"] == "paragraph"
    else:
        assert unit["block_type"] == "paragraph"
        assert unit["basis"] == ("source" if case == "source_api_prose" else "inference")
        assert unit["cited_sources"][0]["kind"] == "code"
        assert unit["text"] == test_inference_api_identity_live.CLAIMS["actual_bad"]
    assert (CODE_API_IDENTITY_POLICY in question["instructions"]) is expected_policy


def test_code_quote_still_requires_its_source_to_support_the_displayed_text() -> None:
    state = _state("altered_code_quote")
    boundary = DecisionsBoundary(probabilities={"grounded_0": 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary, pass_threshold=0.8)(state)

    assert updates["retry"].needs_retry
    assert "b0.content" in updates["retry"].retrieval_feedback
    assert state["response"].result.content.blocks[0].content.text == "    result = np.array([9])"
