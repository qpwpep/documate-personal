"""Planner dialogue keeps native roles and binds exact text from the original ledger."""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.graph_state import PendingAction
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import RequestContract, UserTurnSnapshot, WireRequestContract
from src.runtime.nodes.planner import make_planner_node
from tests.core.helpers import structured_model


@pytest.mark.parametrize("shared_pending_answer", [True, False])
def test_planner_keeps_native_dialogue_and_binds_the_original_input(shared_pending_answer):
    payload = "CANONICAL_INPUT_782\n  keep  these spaces\n"
    original = "Use this original text:\n" + payload
    current = "Copy the original payload exactly. CURRENT_REQUEST_482"
    previous_text = "1. PREVIOUS_OPTION_139\n2. preserve numbered references"
    pending_text = previous_text if shared_pending_answer else "PENDING_BODY_957"
    previous = finalize_answer(text_document(previous_text), [])
    pending = PendingAction(
        contract=RequestContract(request_id="pending-request"),
        response=finalize_answer(text_document(pending_text), []),
        phase="awaiting_destination", body_prepared=True,
    )
    state = build_graph_state_input(
        user_input=current, current_turn_id="current",
        user_turns=(UserTurnSnapshot(turn_id="original", text=original),
                    UserTurnSnapshot(turn_id="current", text=current)),
        messages=[HumanMessage(id="original", content="TRIMMED_DIALOGUE_695"),
                  AIMessage(content=previous_text), HumanMessage(id="current", content=current)],
        previous_response=previous, pending_action=pending,
    )
    observed = []

    def respond(messages):
        observed.extend(messages)
        return InitialPlannerOutput(
            use_retrieval=False, tasks=[], request_contract=WireRequestContract(
                slack_recipient={"state": "omitted"},
                body={"kind": "copy_input", "source": {
                    "turn_id": "original", "quote": payload, "occurrence": None,
                }},
            ),
        )

    result = make_planner_node(structured_model(respond, InitialPlannerOutput), False)(state)

    assert isinstance(observed[-1], HumanMessage)
    assert observed[-1].name is None
    assert observed[-1].content == current
    assert [(type(message), message.content) for message in observed[-3:]] == [
        (HumanMessage, "TRIMMED_DIALOGUE_695"), (AIMessage, previous_text), (HumanMessage, current),
    ]
    context = json.loads(next(message.content for message in observed
                              if message.name == "request_context").split("\n", 1)[1])
    assert context["user_turn_ledger"] == [
        {"turn_id": "original", "text": original},
        {"turn_id": "current", "text": current},
    ]
    offered = {item["ref"]: item for item in context["offered_answer_references"]}
    assert offered["previous"]["document"] == previous.content.model_dump(mode="json")
    assert offered["pending"]["document"] == pending.response.content.model_dump(mode="json")
    assert context["previous_answer_for_aspects"] == previous_text
    assert result["runtime"].request_contract.body.source.text == payload
    assert state["runtime"].user_turns[0].text == original
    assert state["messages"][0].content == "TRIMMED_DIALOGUE_695"


@pytest.mark.parametrize("visible_followup", [None, "The requested translation failed.", "Which answer should I translate?"])
def test_planner_preserves_displayed_followups_without_offering_them_as_answers(visible_followup):
    previous = finalize_answer(text_document("The committed source answer."), [])
    query = "Copy my current request exactly."
    history = [HumanMessage(id="first", content="Prepare the source."), AIMessage(content="The committed source answer.")]
    if visible_followup is not None:
        history.extend([HumanMessage(id="followup", content="Translate it."), AIMessage(content=visible_followup)])
    history.append(HumanMessage(id="current", content=query))
    state = build_graph_state_input(user_input=query, current_turn_id="current", messages=history,
                                    previous_response=previous)
    observed = []

    def respond(messages):
        observed.extend(messages)
        return InitialPlannerOutput(use_retrieval=False, tasks=[], request_contract=WireRequestContract(
            slack_recipient={"state": "omitted"},
            body={"kind": "copy_input", "source": {"turn_id": "current", "quote": query, "occurrence": None}},
        ))

    result = make_planner_node(structured_model(respond, InitialPlannerOutput), False)(state)

    assert [(type(message), message.content) for message in observed[-len(history):]] == [
        (type(message), message.content) for message in history
    ]
    context = json.loads(next(message.content for message in observed
                              if message.name == "request_context").split("\n", 1)[1])
    assert context["offered_answer_references"][0]["document"] == previous.content.model_dump(mode="json")
    assert result["runtime"].request_contract.body.source.text == query


def test_current_input_stays_bound_to_the_raw_ledger_during_repair():
    payload = "  CURRENT_PAYLOAD_614\nunchanged  spacing\n"
    query = "Copy the following text exactly:\n" + payload
    state = build_graph_state_input(
        user_input=query, current_turn_id="current",
        user_turns=(UserTurnSnapshot(turn_id="current", text=query),),
        messages=[HumanMessage(id="current", content=query)],
    )
    observed = []

    def respond(messages):
        observed.append(messages)
        contract = ({
            "body": {"kind": "acknowledge", "evidence_ids": ["instruction"]},
            "evidence": [{"id": "instruction", "turn_id": "unknown-turn", "quote": query,
                          "scope": "current_request", "interpretation": "instruction"}],
        } if len(observed) == 1 else {
            "body": {"kind": "copy_input", "source": {
                "turn_id": "current", "quote": payload, "occurrence": None,
            }},
        })
        return InitialPlannerOutput(
            use_retrieval=False, tasks=[],
            request_contract=WireRequestContract(slack_recipient={"state": "omitted"}, **contract),
        )

    result = make_planner_node(structured_model(respond, InitialPlannerOutput), False)(state)

    assert len(observed) == 2
    for messages in observed:
        original = [message for message in messages if isinstance(message, HumanMessage) and message.content == query]
        assert len(original) == 1
        assert original[0].name is None
        context = json.loads(next(message.content for message in messages
                                  if message.name == "request_context").split("\n", 1)[1])
        assert context["user_turn_ledger"] == [{"turn_id": "current", "text": query}]
        assert context["user_turn_ledger"][0]["text"].encode("utf-8") == original[0].content.encode("utf-8")
        assert sum(str(message.content).count("CURRENT_PAYLOAD_614") for message in messages) == 2
    assert observed[1][-1].content != query
    assert result["runtime"].request_contract.body.source.text == payload
