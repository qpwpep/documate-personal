"""Requested answer positions cannot silently become the latest answer."""
import json

import pytest

from src.core.answer_schema import finalize_answer, text_document
from src.core.request_contracts import WireRequestContract
from src.runtime.agent_runtime.execution_runner import ExecutionRunner
from src.runtime.agent_runtime.session_context import SessionContext
from tests.core.test_planner_request_contract import plan
from tests.core.test_request_contract_retries import _SaveConversation


@pytest.mark.parametrize("requested_ordinal", [1, 3])
def test_planner_binds_only_the_answer_at_the_requested_position(requested_ordinal):
    previous = finalize_answer(text_document("마이그레이션 점검은 아직 시작되지 않았습니다."), [])
    query = f"이 대화에서 {requested_ordinal}번째로 작성한 답변을 저장해줘."
    proposal = WireRequestContract.model_validate({
        "slack_recipient": {"state": "omitted"},
        "body": {"kind": "copy_answer", "source": {"ref": "ordinal", "ordinal": requested_ordinal}},
        "actions": {"save_text": {"intent": "requested", "evidence_ids": ["save"]}},
        "evidence": [{"id": "save", "turn_id": "current", "quote": query,
                      "scope": "current_request", "interpretation": "instruction"}],
    })

    updates, model = plan(query, proposal, previous_response=previous, previous_answer_ordinal=1)

    context = json.loads(next(message.content for message in model.calls[0]
                              if message.name == "request_context").split("\n", 1)[1])
    assert context["offered_answer_references"][0]["ordinal"] == 1
    contract = updates["runtime"].request_contract
    if requested_ordinal == 1:
        assert contract.body.kind == "copy_answer"
        assert contract.body.source.response_hash == previous.content_hash
        assert contract.body_request.source.ordinal == 1
        assert contract.can_prepare_body()
    else:
        assert contract.body.kind == "unresolved"
        assert not contract.execution_ready("save_text", body_ready=True)
        assert {gap.slot for gap in contract.missing_info} == {"answer_reference"}
        assert "순서" in contract.clarification_question


def test_displayed_answer_positions_count_equal_bodies_and_ignore_followups():
    session = SessionContext()
    answer = finalize_answer(text_document("같은 본문"), [])
    for expected in [1, 2]:
        session.commit_response_state(response=answer, response_kind="answer", pending_request=None)
        assert session.previous_answer_ordinal == expected
    session.commit_response_state(response=answer, response_kind="failure", pending_request=None)
    assert session.previous_answer_ordinal == 2
    session.reset_conversation_memory()
    assert session.previous_answer_ordinal is None
    session.commit_response_state(response=answer, response_kind="answer", pending_request=None)
    assert session.previous_answer_ordinal == 1


def test_nonexistent_answer_never_reaches_synthesis_or_file_delivery(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    session = SessionContext()
    session.commit_previous_response(finalize_answer(text_document("첫 번째 답변"), []))
    conversation = _SaveConversation([])
    conversation.payload = {"body": {"kind": "copy_answer", "source": {"ref": "ordinal", "ordinal": 3}}}
    runner = ExecutionRunner(graph=conversation.graph, session=session)

    result, _elapsed = runner.invoke_graph(runner.prepare_graph_state("세 번째 답변을 저장해줘."))

    assert result["runtime"].previous_answer_ordinal == 1
    assert result["runtime"].request_contract.body.kind == "unresolved"
    assert result["response"].kind == "clarification"
    assert result["response"].result.actions == []
    assert list(tmp_path.iterdir()) == []
