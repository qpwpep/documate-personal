"""Local sentence quotas control repair and saving despite unanimous model approval."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import BaseMessage
from openai import OpenAI

from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.decisions import DecisionsClient
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.make_graph import build_graph
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.session import add_user_message, make_summarize_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.helpers import mock_chat_model, structured_model
from tests.synthesis_fixtures import synthesis_excerpt_limits


ORIGINAL = "민수는 도서관에 갔다. 민수는 책을 빌렸다. 지수는 집에서 공부했다."
REJECTED = "민수는 도서관을 방문했다. 민수는 책을 대출했다. 지수는 집에서 공부했다."
REPAIRED = "민수는 도서관에서 책을 빌렸다. 지수는 집에서 공부했다."
CONDITION = "민수와 지수의 행동을 설명한다."
QUERY = "방금 답변을 요약해줘. 민수와 지수의 행동을 각각 한 문장으로 설명해줘. 완성된 답변을 TXT로 저장해줘. Slack으로 보내지 마."


def _planner_reply(messages: list[BaseMessage]) -> InitialPlannerOutput:
    context = next(message for message in messages if message.name == "request_context")
    user = json.loads(context.content.split("\n", 1)[1])["user_turn_ledger"][-1]
    contract = WireRequestContract.model_validate({
        "body": {"kind": "transform_answer", "source": {"ref": "previous"},
                 "instruction": "이전 답변에 등장하는 민수와 지수의 행동을 요약한다.", "evidence_ids": ["request"]},
        "actions": {
            "save_text": {"intent": "requested", "evidence_ids": ["request"]},
            "slack_notify": {"intent": "forbidden", "evidence_ids": ["request"]},
        },
        "answer": {
            "content": [{"kind": "explanation", "mode": "required",
                         "description": CONDITION, "evidence_ids": ["request"]}],
            "sentence_counts": [
                {"scope": scope, "minimum": 1, "maximum": 1, "mode": "required", "evidence_ids": ["request"]}
                for scope in ("민수의 행동", "지수의 행동")
            ],
        },
        "slack_recipient": {"state": "omitted"},
        "evidence": [{"id": "request", "turn_id": user["turn_id"], "quote": user["text"],
                      "scope": "current_request", "interpretation": "instruction"}],
    })
    return InitialPlannerOutput(use_retrieval=False, tasks=[], request_contract=contract)


def _unavailable_boundary(*args: object, **kwargs: object) -> None:
    raise AssertionError("This request must neither retrieve external sources nor send Slack messages")


def _run_request(
    output: Path, monkeypatch: pytest.MonkeyPatch, *, repaired: bool,
) -> list[GraphState]:
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    drafts = iter((False, repaired))

    def synthesize(_messages: list[BaseMessage]) -> object:
        minsu = ("민수는 도서관에서 책을 빌렸다." if next(drafts)
                 else "민수는 도서관을 방문했다. 민수는 책을 대출했다.")
        return AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [
            {"text": minsu, "basis": "interaction", "refs": [], "sentence_scope": 1},
            {"text": "지수는 집에서 공부했다.", "basis": "interaction", "refs": [], "sentence_scope": 2},
        ]}]})

    def judge(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        # No model verdict detects the bad draft. Its actual sentence quota must.
        answers = [{"type": "predicate", "name": question["name"], "probability": 0.99}
                   for question in payload["questions"]]
        return httpx.Response(200, headers={"x-request-id": "sentence-condition-judge"}, json={
            "model": "gpt-6-luna", "answers": answers,
            "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32},
        })

    policy = ConversationMemoryPolicy()
    with OpenAI(api_key="test-key", base_url="https://provider.test/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(judge))) as sdk:
        decisions = DecisionsClient(sdk)
        graph = build_graph(
            state_type=GraphState, add_user_node=add_user_message,
            summarize_node=make_summarize_node(None, False, policy=policy),
            planner_node=make_planner_node(structured_model(_planner_reply, InitialPlannerOutput), False),
            retrieve_dispatch_node=make_retrieve_dispatch_node(_unavailable_boundary, _unavailable_boundary, False),
            synthesize_node=make_synthesize_node(mock_chat_model(synthesize)),
            pre_synthesis_validation_node=make_pre_synthesis_validation_node(
                False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=decisions,
            ),
            post_synthesis_validation_node=make_post_synthesis_validation_node(False, decisions_client=decisions),
            action_postprocess_node=make_action_postprocess_node(build_save_text_tool(), _unavailable_boundary, False),
            memory_policy=policy,
        )
        state = build_graph_state_input(
            user_input=QUERY, previous_response=finalize_answer(text_document(ORIGINAL), []),
            retry=RetryState(max_retries=0, max_content_repairs=1),
        )
        snapshots = []
        for snapshot in graph.stream(state, stream_mode="values"):
            if "response" not in snapshot or snapshot["response"].kind == "draft":
                assert list(output.glob("*.txt")) == []
            snapshots.append(snapshot)
        return snapshots


def test_rejected_sentence_condition_repairs_body_before_saving(tmp_path, monkeypatch):
    snapshots = _run_request(tmp_path, monkeypatch, repaired=True)
    result = snapshots[-1]
    answer = result["response"].result

    assert export_answer_text(answer) == REPAIRED
    assert result["response"].kind == "answer"
    assert result["response"].synthesis_attempt == 2
    contract = result["runtime"].request_contract
    confirmed = [snapshot["runtime"].request_contract for snapshot in snapshots
                 if snapshot["runtime"].request_contract is not None]
    assert all(item == contract for item in confirmed)
    assert contract.body.source.response_hash == finalize_answer(text_document(ORIGINAL), []).content_hash
    assert contract.answer.content[0].description == CONDITION
    assert [(condition.scope, condition.minimum, condition.maximum) for condition in contract.answer.sentence_counts] == [
        ("민수의 행동", 1, 1), ("지수의 행동", 1, 1),
    ]
    assert contract.answer.format == ()
    assert contract.actions.save_text.intent == "requested"
    assert contract.actions.slack_notify.intent == "forbidden"
    assert result["response"].request_id == contract.request_id
    assert result["response"].contract_revision == contract.revision
    assert any("sentence_count" in snapshot["retry"].retrieval_feedback for snapshot in snapshots)
    condition = next(check for check in answer.semantic_validation.checks if check.name == "content_condition:0")
    assert condition.status == "evaluated" and condition.probability == 0.99
    assert answer.semantic_validation.passed
    assert [(receipt.kind, receipt.status) for receipt in answer.actions] == [("save_text", "success")]
    files = list(tmp_path.glob("*.txt"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8-sig") == REPAIRED


def test_sentence_condition_failure_after_repair_does_not_publish_or_save_the_source(tmp_path, monkeypatch):
    result = _run_request(tmp_path, monkeypatch, repaired=False)[-1]

    assert result["response"].kind == "failure"
    assert result["response"].problem.code == "answer_validation_failed"
    assert result["response"].synthesis_attempt == 2
    assert result["retry"].content_repair_attempt == 1
    assert not result["retry"].needs_retry
    assert export_answer_text(result["response"].result) not in {ORIGINAL, REJECTED}
    assert result["response"].result.actions == []
    assert list(tmp_path.glob("*.txt")) == []
    assert result["runtime"].pending_action.body_prepared is False
