"""Editing obligations survive planner binding and pending-request reconstruction."""
import pytest
from pydantic import ValidationError

from src.core.answer_schema import finalize_answer, text_document
from src.core.request_bodies import body_to_wire
from src.core.request_contracts import RequestContract, WireRequestContract
from tests.core.test_planner_request_contract import plan


@pytest.mark.parametrize("kind", ["transform_input", "transform_answer"])
@pytest.mark.parametrize("change_requirement", ["required", "if_needed"])
def test_planner_preserves_the_explicit_change_requirement(kind, change_requirement):
    original = "자료를 다음 주 월요일까지 제출해 주시기 바랍니다."
    query = f'이 문장을 더 공손하게 바꿔줘: "{original}"'
    source = ({"turn_id": "current", "quote": original, "occurrence": None}
              if kind == "transform_input" else {"ref": "previous"})
    proposal = WireRequestContract.model_validate({
        "slack_recipient": {"state": "omitted"},
        "body": {"kind": kind, "source": source, "instruction": "Preserve the deadline and revise the tone.",
                 "change_requirement": change_requirement, "evidence_ids": ["edit"]},
        "evidence": [{"id": "edit", "turn_id": "current", "quote": query,
                      "scope": "body", "interpretation": "instruction"}],
    })

    updates, _ = plan(query, proposal, previous_response=finalize_answer(text_document(original), []))

    contract = updates["runtime"].request_contract
    assert contract.body.change_requirement == change_requirement
    assert contract.body_request.change_requirement == change_requirement
    assert body_to_wire(contract.body).change_requirement == change_requirement


@pytest.mark.parametrize("kind", ["transform_input", "transform_answer"])
def test_model_transform_contract_requires_a_change_requirement(kind):
    source = ({"turn_id": "current", "quote": "source", "occurrence": None}
              if kind == "transform_input" else {"ref": "previous"})
    with pytest.raises(ValidationError, match="change_requirement"):
        WireRequestContract.model_validate({
            "slack_recipient": {"state": "omitted"},
            "body": {"kind": kind, "source": source, "instruction": "Edit the source."},
        })


def test_bound_transform_contract_requires_a_change_requirement():
    previous = finalize_answer(text_document("source"), [])
    with pytest.raises(ValidationError, match="change_requirement"):
        RequestContract.model_validate({
            "body": {"kind": "transform_answer", "source": {
                "ref": "previous", "response_hash": previous.content_hash,
            }, "instruction": "Edit the source."},
        })
