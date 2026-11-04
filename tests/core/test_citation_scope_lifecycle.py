"""Citation restrictions survive request binding, pending turns and generation."""
import json

from src.core.answer_schema import finalize_answer, text_document
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.pending_request import PendingRequest
from src.core.request_contracts import UserTurnSnapshot, WireRequestContract
from src.runtime.nodes.synthesis import make_synthesize_node
from tests.core.helpers import mock_chat_model
from tests.core.test_planner_request_contract import plan, state_for
from tests.synthesis_fixtures import prepare_synthesis_state


def _initial_request(*, kind="code"):
    clause = "코드에서만 인용해줘" if kind == "code" else "텍스트에서만 인용해줘"
    query = f"설명을 준비하고 {clause}. Slack으로 보내줘."
    proposal = WireRequestContract.model_validate({
        "slack_recipient": {"state": "omitted"},
        "body": {"kind": "compose", "instruction": "설명을 준비한다.", "evidence_ids": []},
        "actions": {"slack_notify": {"intent": "requested", "evidence_ids": ["delivery"]}},
        "answer": {"citation_scope": {"allowed_kinds": [kind], "evidence_ids": ["scope"]}},
        "evidence": [
            {"id": "scope", "turn_id": "old", "quote": clause,
             "scope": "answer.citation_scope", "interpretation": "instruction"},
            {"id": "delivery", "turn_id": "old", "quote": "Slack으로 보내줘.",
             "scope": "actions.slack_notify", "interpretation": "instruction"},
        ],
    })
    turns = (UserTurnSnapshot(turn_id="old", text=query),)
    updates, _ = plan(query, proposal, current_turn_id="old", user_turns=turns)
    return query, turns, updates


def test_bound_citation_scope_reaches_synthesis_without_planner_evidence_ids():
    query, turns, updates = _initial_request()
    contract = updates["runtime"].request_contract
    scope = contract.answer.citation_scope
    assert scope.allowed_kinds == ("code",)
    bound_id, = scope.evidence_ids
    assert bound_id != "scope"
    assert next(item for item in contract.evidence if item.id == bound_id).quote == "코드에서만 인용해줘"
    observed = []

    def respond(messages):
        observed.extend(messages)
        return text_document("설명을 준비했습니다.").model_dump(mode="json")

    state = state_for(query, current_turn_id="old", user_turns=turns)
    state.update(updates)
    prepare_synthesis_state(state)
    make_synthesize_node(mock_chat_model(respond))(state)

    prompt = next(message.content for message in observed if "[Confirmed Request Contract]" in message.content)
    record = next(json.loads(line) for line in prompt.splitlines() if line.startswith('{"'))
    assert record["answer"]["citation_scope"] == {"allowed_kinds": ["code"]}
    assert bound_id not in prompt


def test_pending_supplement_preserves_citation_scope_and_its_original_evidence():
    _query, turns, updates = _initial_request()
    original = updates["runtime"].request_contract
    answer = finalize_answer(text_document("준비된 답변"), [])
    pending = PendingRequest(contract=original, response=answer,
                             phase="awaiting_destination", body_prepared=True)
    query = "준비된 답변을 그대로 이어서 처리해줘."
    proposal = WireRequestContract.model_validate({
        "relation": "supplement", "target_request_id": original.request_id,
        "slack_recipient": {"state": "omitted"},
        "body": {"kind": "copy_answer", "source": {"ref": "pending"}},
    })

    result, _ = plan(query, proposal, pending_request=pending,
                     user_turns=(*turns, UserTurnSnapshot(turn_id="current", text=query)))

    contract = result["runtime"].request_contract
    assert contract.answer.citation_scope == original.answer.citation_scope
    evidence = {item.id: item for item in contract.evidence}
    for identifier in original.answer.citation_scope.evidence_ids:
        assert evidence[identifier] == next(item for item in original.evidence if item.id == identifier)
    assert result["runtime"].pending_request.body_prepared
    assert contract.body.source.response_hash == answer.content_hash


def test_current_scope_correction_replaces_the_restriction_and_invalidates_prepared_body():
    _query, turns, updates = _initial_request(kind="paragraph")
    original = updates["runtime"].request_contract
    snapshot = build_snapshot(
        source_uri="upload:///notes.md", title="notes.md", media_type="text/markdown",
        source_type="upload", content="먼저 준비한 답변", parser="test", parser_version="1",
    )
    source = build_evidence(
        snapshot=snapshot,
        element=DocumentElement(element_id="content", kind="paragraph", text="먼저 준비한 답변"),
    )
    answer = finalize_answer(text_document("먼저 준비한 답변", basis="source", refs=[source.id]), [source])
    pending = PendingRequest(contract=original, response=answer,
                             phase="awaiting_destination", body_prepared=True)
    query = "정정할게. 이제 코드에서만 인용해줘."
    proposal = WireRequestContract.model_validate({
        "relation": "correction", "target_request_id": original.request_id,
        "slack_recipient": {"state": "omitted"},
        "body": {"kind": "copy_answer", "source": {"ref": "pending"}},
        "answer": {"citation_scope": {"allowed_kinds": ["code"], "evidence_ids": ["change"]}},
        "evidence": [{"id": "change", "turn_id": "current", "quote": "코드에서만 인용해줘",
                      "scope": "answer.citation_scope", "interpretation": "correction"}],
    })

    result, _ = plan(query, proposal, pending_request=pending,
                     user_turns=(*turns, UserTurnSnapshot(turn_id="current", text=query)))

    contract = result["runtime"].request_contract
    assert contract.answer.citation_scope.allowed_kinds == ("code",)
    assert original.answer.citation_scope.allowed_kinds == ("paragraph",)
    changed_id, = contract.answer.citation_scope.evidence_ids
    assert changed_id != "change"
    evidence = {item.id: item for item in contract.evidence}
    assert evidence[changed_id].turn_id == "current"
    assert not set(original.answer.citation_scope.evidence_ids).intersection(evidence)
    assert not result["runtime"].pending_request.body_prepared
    assert result["runtime"].pending_request.phase == "awaiting_body"
