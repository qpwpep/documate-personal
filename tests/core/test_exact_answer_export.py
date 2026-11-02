"""An explicit export preserves the displayed answer instead of regenerating it."""
import json
from functools import partial
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.answer_export import answer_export_key, answer_export_snapshot, build_answer_export_operation
from src.core.answer_schema import ResponseIssue, export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.planner_schema import InitialPlannerOutput
from src.infra import saved_artifacts
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.make_graph import build_graph
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.session import add_user_message, make_summarize_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.helpers import SequentialGraph, make_decisions_client, mock_chat_model, structured_model
from tests.core.test_decisions_validation import _state
from tests.core.test_pending_action_delivery import _manager
from tests.synthesis_fixtures import synthesis_excerpt_limits


def _unavailable(*args, **kwargs):
    raise AssertionError("An unchanged export does not generate, retrieve, or send to Slack")


class ExportPlanner:
    def __init__(self, *, semantic_constraint=False, transform=False, action="save_text"):
        self.semantic_constraint = semantic_constraint
        self.transform = transform
        self.action = action

    def invoke(self, messages):
        context = next(message for message in messages if message.name == "request_context")
        user = json.loads(context.content.split("\n", 1)[1])["user_turn_ledger"][-1]
        return InitialPlannerOutput(use_retrieval=False, tasks=[], request_contract={
            "body": ({"kind": "transform_answer", "source": {"ref": "previous"},
                      "instruction": "이전 답변을 요약해줘", "evidence_ids": ["export"]} if self.transform else
                     {"kind": "copy_answer", "source": {"ref": "previous"}}),
            "actions": {self.action: {"intent": "requested", "evidence_ids": ["export"]}} if self.action else {},
            "answer": {"content": [{"kind": "custom", "mode": "required", "description": "새 조건",
                                     "evidence_ids": ["export"]}]} if self.semantic_constraint else {},
            "evidence": [{"id": "export", "turn_id": user["turn_id"], "quote": user["text"],
                          "scope": "current_request", "interpretation": "instruction"}],
            "slack_recipient": {"state": "omitted"},
        }).model_dump(mode="json")


def export_graph(*, semantic_constraint=False, probabilities=None, post_probabilities=None, transform=False, action="save_text"):
    policy = ConversationMemoryPolicy()
    planner = ExportPlanner(semantic_constraint=semantic_constraint, transform=transform, action=action)
    synthesizer = mock_chat_model(
        (lambda _messages: text_document("원문이 보장하지 않는 새 설명입니다.", basis="source", refs=["e1"]).model_dump(mode="json"))
        if transform else _unavailable
    )
    return build_graph(
        GraphState, add_user_message, make_summarize_node(None, False, policy=policy),
        make_planner_node(structured_model(planner.invoke, InitialPlannerOutput), False),
        make_retrieve_dispatch_node(_unavailable, _unavailable, False),
        make_synthesize_node(synthesizer), memory_policy=policy,
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(
            False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client(probabilities)),
        post_synthesis_validation_node=make_post_synthesis_validation_node(
            False, decisions_client=make_decisions_client(
                {"request_fulfilled": 0.01} if post_probabilities is None else post_probabilities)),
        action_postprocess_node=make_action_postprocess_node(build_save_text_tool(), _unavailable, False),
    )


def checked_source():
    state = _state()
    result = make_post_synthesis_validation_node(False, decisions_client=make_decisions_client())(state)["response"].result
    return result.model_copy(update={"issues": [ResponseIssue(code="source_limit", message="추가 입력의 결과는 확인하지 않았습니다.")]})


@pytest.mark.parametrize("export", [
    answer_export_snapshot, answer_export_key, export_answer_text,
    partial(build_answer_export_operation, session_id="exact-export-session"),
])
def test_export_boundaries_reject_body_changed_after_validation(export):
    source = finalize_answer(text_document("검증한 본문"), [])
    source.content = text_document("검증 후 바뀐 본문")

    with pytest.raises(ValidationError, match="content changed"):
        export(source)


def test_export_operation_preserves_literal_body_and_limit_bytes():
    source = finalize_answer(text_document("첫 문장입니다.\n둘째 문장입니다."), [], issues=[
        ResponseIssue(code="source_limit", message="첨부된 자료만 확인했습니다."),
    ])

    operation = build_answer_export_operation(source, session_id="exact-export-session")

    expected = "첫 문장입니다.\n둘째 문장입니다.\n\n참고 및 제한\n- 첨부된 자료만 확인했습니다."
    assert operation.matches_bytes(expected.encode("utf-8-sig"))


def test_export_preserves_checked_body_sources_limits_and_semantic_report(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    source = checked_source()
    result = export_graph().invoke(build_graph_state_input(
        user_input="방금 답변을 출처와 함께 txt 파일로 저장해줘", previous_response=source,
        session_id="exact-export-session",
    ))

    assert result["response"].kind == "answer"
    answer = result["response"].result
    assert answer.model_dump(exclude={"actions"}) == source.model_dump(exclude={"actions"})
    receipt = answer.actions[0]
    assert receipt.status == "success"
    assert Path(receipt.file_path).read_bytes() == export_answer_text(source, include_sources=True).encode("utf-8-sig")


def test_unchanged_export_retry_and_repeat_use_one_exact_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    source = checked_source()
    graph = export_graph()
    real_link = saved_artifacts.os.link

    def fail_payload(source, destination):
        if Path(destination).suffix == ".txt":
            raise OSError("temporary disk failure")
        return real_link(source, destination)

    state = dict(user_input="방금 답변을 출처와 함께 txt 파일로 저장해줘", previous_response=source,
                 session_id="exact-export-session")
    with monkeypatch.context() as failing:
        failing.setattr(saved_artifacts.os, "link", fail_payload)
        first = graph.invoke(build_graph_state_input(**state))
    assert first["response"].result.actions[0].status == "error"
    for _ in range(2):
        result = graph.invoke(build_graph_state_input(**state))
        receipt = result["response"].result.actions[0]
        assert receipt.status == "success"
        assert receipt.operation == first["response"].result.actions[0].operation
        assert Path(receipt.file_path).read_bytes() == export_answer_text(source, include_sources=True).encode("utf-8-sig")
    assert len(list(tmp_path.glob("*.txt"))) == 1


def test_copy_with_new_semantic_requirement_still_requires_semantic_validation(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    result = export_graph(semantic_constraint=True).invoke(build_graph_state_input(
        user_input="방금 답변이 새 조건을 충족하면 출처와 함께 저장해줘", previous_response=checked_source(),
        session_id="constrained-export-session",
    ))
    assert result["response"].kind == "failure"
    assert result["response"].result.actions == []
    assert list(tmp_path.glob("*.txt")) == []


def test_revalidated_copy_reuses_the_original_save_operation_and_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    source = checked_source()
    manager = _manager(SequentialGraph(
        export_graph(),
        export_graph(semantic_constraint=True, post_probabilities={"request_fulfilled": 0.95}),
    ))
    manager.session.previous_response = source

    first = manager.run_agent_flow("방금 답변을 출처와 함께 저장해줘")
    repeated = manager.run_agent_flow("방금 답변이 새 조건을 충족하면 출처와 함께 저장해줘")

    assert first["status"] == repeated["status"] == "completed"
    assert first["response"]["semantic_validation"] != repeated["response"]["semantic_validation"]
    first_receipt = first["response"]["actions"][0]
    repeated_receipt = repeated["response"]["actions"][0]
    assert repeated_receipt["operation"] == first_receipt["operation"]
    assert repeated_receipt["artifact"] == first_receipt["artifact"]
    assert Path(repeated_receipt["file_path"]).read_bytes() == export_answer_text(
        source, include_sources=True).encode("utf-8-sig")
    assert len(list(tmp_path.glob("*.txt"))) == 1


def test_export_does_not_rejudge_whether_existing_body_can_be_prepared(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    source = checked_source()
    result = export_graph(probabilities={"request_ready": 0.01, "compact_request_ready": 0.01}).invoke(
        build_graph_state_input(user_input="방금 답변을 저장해줘", previous_response=source,
                                session_id="ready-export-session"))
    assert result["response"].kind == "answer"
    assert result["response"].result.actions[0].status == "success"


def test_export_failure_feedback_does_not_replace_target_of_a_followup_retry(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    source = checked_source()
    manager = _manager(SequentialGraph(export_graph(semantic_constraint=True), export_graph()))
    manager.session.previous_response = source.model_copy(deep=True)

    failed = manager.run_agent_flow("방금 답변이 새 조건을 충족하면 출처와 함께 저장해줘")

    assert failed["status"] == "failed"
    assert failed["answer_id"] is None
    assert manager.session.previous_response == source
    assert manager.session.pending_action is None
    assert list(tmp_path.glob("*.txt")) == []

    retried = manager.run_agent_flow("방금 답변을 출처와 함께 txt 파일로 저장해줘")
    assert retried["status"] == "completed"
    assert retried["answer_id"] is not None
    receipt = retried["response"]["actions"][0]
    assert Path(receipt["file_path"]).read_bytes() == export_answer_text(source, include_sources=True).encode("utf-8-sig")


@pytest.mark.parametrize("action, prompt", [
    ("save_text", "방금 답변을 요약해서 출처와 함께 저장해줘"),
    ("slack_notify", "방금 답변을 요약해서 출처와 함께 Slack으로 보내줘"),
    (None, "방금 답변을 요약해서 출처와 함께 보여줘"),
])
def test_failed_transformation_keeps_original_answer_as_retry_target(tmp_path, monkeypatch, action, prompt):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    source = checked_source()
    manager = _manager(export_graph(transform=True, action=action))
    manager.session.previous_response = source.model_copy(deep=True)
    failed = manager.run_agent_flow(prompt)
    assert failed["status"] == "partial"
    assert failed["answer_id"] is None
    assert manager.session.previous_response == source
    assert manager.session.pending_action.response == source
    assert list(tmp_path.glob("*.txt")) == []
