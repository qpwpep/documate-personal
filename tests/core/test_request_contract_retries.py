from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.tools.save_text import build_save_text_tool
from src.infra import saved_artifacts
from src.runtime.make_graph import build_graph
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.session import add_user_message, make_summarize_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node


class PlannerBoundary:
    def __init__(self, previous, save_intent):
        self.previous = previous
        self.save_intent = save_intent

    def invoke(self, messages):
        user = [message for message in messages if isinstance(message, HumanMessage)][-1]
        contract = WireRequestContract.model_validate({
            "body": {"kind": "transform_answer", "source": {"ref": "previous"},
                     "instruction": "세 줄로 줄여줘", "evidence_ids": ["format"]},
            "actions": {"save_text": {"intent": self.save_intent, "evidence_ids": ["action"]}},
            "answer": {"format": [{"kind": "line_count", "mode": "required", "value": 3, "evidence_ids": ["format"]}]},
            "evidence": [
                {"id": "format", "turn_id": user.id or "user:0", "quote": str(user.content), "scope": "answer.format.line_count", "interpretation": "instruction"},
                {"id": "action", "turn_id": user.id or "user:0", "quote": str(user.content), "scope": "actions.save_text",
                 "interpretation": "instruction" if self.save_intent == "requested" else "negation"},
            ],
        })
        return PlannerOutput(use_retrieval=False, tasks=[], request_contract=contract)


class RepairingModelBoundary:
    def __init__(self):
        self.outputs = iter(["하나\n둘\n셋\n넷", "첫 요약\n둘째 요약\n셋째 요약"])

    def invoke(self, _messages):
        return text_document(next(self.outputs)).model_dump(mode="json")


def unavailable_boundary(*_args, **_kwargs):
    raise AssertionError("This request requires neither external lookup nor Slack delivery")


@pytest.mark.parametrize("save_intent", ["requested", "forbidden"])
def test_transform_repair_preserves_contract_and_delivers_only_the_repaired_body(tmp_path, monkeypatch, save_intent):
    """검색 없는 수정도 한 번 재합성하며 저장 의사와 세 줄 조건을 바꾸지 않는다."""
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    previous = finalize_answer(text_document("이전 첫 줄\n이전 둘째 줄\n이전 셋째 줄\n이전 넷째 줄"), [])
    policy = ConversationMemoryPolicy()
    graph = build_graph(
        state_type=GraphState, add_user_node=add_user_message,
        summarize_node=make_summarize_node(None, False, policy=policy),
        planner_node=make_planner_node(PlannerBoundary(previous, save_intent), False),
        retrieve_dispatch_node=make_retrieve_dispatch_node(unavailable_boundary, unavailable_boundary, False),
        synthesize_node=make_synthesize_node(RepairingModelBoundary()),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False),
        action_postprocess_node=make_action_postprocess_node(build_save_text_tool(), unavailable_boundary, False),
        memory_policy=policy,
    )
    query = "방금 답변을 세 줄로 줄여 저장해줘" if save_intent == "requested" else "방금 답변을 세 줄로 줄이고 저장하지 마"
    result = graph.invoke(build_graph_state_input(user_input=query, previous_response=previous, retry=RetryState(max_retries=2)))
    answer = result["response"].result
    expected = "첫 요약\n둘째 요약\n셋째 요약"
    assert export_answer_text(answer) == expected
    assert result["response"].kind == "answer"
    assert result["response"].synthesis_attempt == 2
    assert result["runtime"].request_contract.to_wire() == result["planner"].output.request_contract
    assert result["runtime"].request_contract.actions.save_text.intent == save_intent
    files = list(tmp_path.glob("*.txt"))
    if save_intent == "requested":
        assert len(files) == 1
        assert files[0].read_text(encoding="utf-8-sig") == expected
    else:
        assert files == []
        assert answer.actions == []


def test_failed_save_retries_the_same_frozen_operation_through_real_planner(tmp_path, monkeypatch):
    """A new user turn resumes the same saved body through the production graph."""
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)

    class DeliveryPlannerBoundary:
        pending_request_id = None

        def invoke(self, messages):
            user = [message for message in messages if isinstance(message, HumanMessage)][-1]
            payload = {
                "actions": {"save_text": {"intent": "requested", "evidence_ids": ["save"]}},
                "evidence": [{"id": "save", "turn_id": user.id, "quote": str(user.content),
                              "scope": "current_request", "interpretation": "instruction"}],
            }
            if self.pending_request_id is None:
                payload["body"] = {"kind": "compose", "instruction": "설명을 작성한다.", "evidence_ids": ["save"]}
            else:
                payload.update(relation="supplement", target_request_id=self.pending_request_id,
                               body={"kind": "copy_answer", "source": {"ref": "pending"}})
            return PlannerOutput(use_retrieval=False, tasks=[],
                                 request_contract=WireRequestContract.model_validate(payload))

    planner_boundary = DeliveryPlannerBoundary()
    policy = ConversationMemoryPolicy()
    graph = build_graph(
        state_type=GraphState, add_user_node=add_user_message,
        summarize_node=make_summarize_node(None, False, policy=policy),
        planner_node=make_planner_node(planner_boundary, False),
        retrieve_dispatch_node=make_retrieve_dispatch_node(unavailable_boundary, unavailable_boundary, False),
        synthesize_node=make_synthesize_node(RepairingModelBoundary()),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False),
        action_postprocess_node=make_action_postprocess_node(build_save_text_tool(), unavailable_boundary, False),
        memory_policy=policy,
    )
    real_link = saved_artifacts.os.link

    def fail_payload_publication(source, destination):
        if Path(destination).suffix == ".txt":
            raise OSError("temporary storage failure")
        return real_link(source, destination)

    with monkeypatch.context() as failure:
        failure.setattr(saved_artifacts.os, "link", fail_payload_publication)
        first = graph.invoke(build_graph_state_input(
            user_input="설명을 작성하고 파일로 저장해줘", session_id="save-retry-session",
        ))
    failed_receipt = first["response"].result.actions[0]
    pending = first["runtime"].pending_action
    assert failed_receipt.status == "error"
    assert failed_receipt.error_code == "write_failed"
    assert pending is not None and pending.body_prepared
    assert pending.phase == "awaiting_delivery"
    assert pending.save_operation == failed_receipt.operation
    assert pending.completed_actions == ()
    assert list(tmp_path.glob("*.txt")) == []
    assert len(list(tmp_path.glob("*.json"))) == 1

    planner_boundary.pending_request_id = pending.contract.request_id
    second = graph.invoke(build_graph_state_input(
        user_input="그 본문 그대로 저장을 다시 시도해줘", session_id="save-retry-session",
        messages=list(first["messages"]), user_turns=first["runtime"].user_turns,
        previous_response=first["response"].result, pending_action=pending,
    ))
    completed_receipt = second["response"].result.actions[0]
    assert second["runtime"].request_contract.revision == pending.contract.revision + 1
    assert second["runtime"].request_contract.body.kind == "copy_answer"
    assert second["runtime"].request_contract.body.source.ref == "pending"
    assert second["runtime"].request_contract.actions.save_text.intent == "requested"
    assert second["response"].result.content == first["response"].result.content
    assert completed_receipt.status == "success"
    assert completed_receipt.operation == failed_receipt.operation
    assert completed_receipt.operation.target_kind == "compose"
    assert second["response"].save_operation_binding_sha256 == failed_receipt.operation.binding_sha256
    assert second["runtime"].pending_action is None
    expected = export_answer_text(first["response"].result, include_sources=True).encode("utf-8-sig")
    assert Path(completed_receipt.file_path).read_bytes() == expected
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


class _SaveConversation:
    """Exercise request reconciliation through the production graph and real files."""

    def __init__(self, documents, *, slack=unavailable_boundary):
        self.documents = iter(documents)
        self.result = None
        self.payload = {}
        policy = ConversationMemoryPolicy()
        self.graph = build_graph(
            state_type=GraphState, add_user_node=add_user_message,
            summarize_node=make_summarize_node(None, False, policy=policy),
            planner_node=make_planner_node(self, False),
            retrieve_dispatch_node=make_retrieve_dispatch_node(unavailable_boundary, unavailable_boundary, False),
            synthesize_node=make_synthesize_node(self._DocumentBoundary(self.documents)),
            pre_synthesis_validation_node=make_pre_synthesis_validation_node(False),
            post_synthesis_validation_node=make_post_synthesis_validation_node(False),
            action_postprocess_node=make_action_postprocess_node(build_save_text_tool(), slack, False),
            memory_policy=policy,
        )

    class _DocumentBoundary:
        def __init__(self, documents):
            self.documents = documents

        def invoke(self, _messages):
            return text_document(next(self.documents)).model_dump(mode="json")

    def invoke(self, messages):
        user = [message for message in messages if isinstance(message, HumanMessage)][-1]
        payload = {
            "actions": {"save_text": {"intent": "requested", "evidence_ids": ["request"]}},
            "evidence": [{"id": "request", "turn_id": user.id, "quote": str(user.content),
                          "scope": "current_request", "interpretation": "instruction"}],
            **self.payload,
        }
        if self.result is not None:
            payload["target_request_id"] = self.result["runtime"].pending_action.contract.request_id
        if callable(payload.get("body")):
            payload["body"] = payload["body"](user.id)
        return PlannerOutput(use_retrieval=False, tasks=[],
                             request_contract=WireRequestContract.model_validate(payload))

    def turn(self, query, **payload):
        self.payload = payload
        previous = self.result
        self.result = self.graph.invoke(build_graph_state_input(
            user_input=query, session_id="save-conversation",
            messages=list(previous["messages"]) if previous else [],
            user_turns=previous["runtime"].user_turns if previous else (),
            previous_response=previous["response"].result if previous else None,
            pending_action=previous["runtime"].pending_action if previous else None,
        ))
        return self.result


def _fail_save_publication(monkeypatch):
    real_link = saved_artifacts.os.link

    def fail_payload_publication(source, destination):
        if Path(destination).suffix == ".txt":
            raise OSError("temporary storage failure")
        return real_link(source, destination)

    monkeypatch.setattr(saved_artifacts.os, "link", fail_payload_publication)


def test_failed_save_body_correction_survives_clarification_before_supplement(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    conversation = _SaveConversation(["원래 본문", "보충 내용을 반영한 새 본문"])
    with monkeypatch.context() as failure:
        _fail_save_publication(failure)
        first = conversation.turn("설명을 작성하고 저장해줘", body={
            "kind": "compose", "instruction": "설명을 작성한다.", "evidence_ids": ["request"],
        })
    original = first["response"].result
    operation = first["runtime"].pending_action.save_operation
    assert original.actions[0].error_code == "write_failed"

    second = conversation.turn("그 본문을 새로운 주제에 맞게 고쳐줘", relation="correction", body={
        "kind": "transform_answer", "source": {"ref": "pending"},
        "instruction": "새 주제에 맞게 본문을 수정한다.", "evidence_ids": ["request"],
    }, missing_info=[{"slot": "subject", "reason": "unclear", "question": "어떤 주제로 바꿀까요?"}])
    pending = second["runtime"].pending_action
    assert second["response"].kind == "clarification"
    assert pending.response.content == original.content
    assert pending.body_prepared is False
    assert pending.save_operation is None
    assert pending.save_receipt is None
    assert list(tmp_path.glob("*.txt")) == []

    third = conversation.turn("새 주제는 재시도야", relation="supplement", body={
        "kind": "transform_answer", "source": {"ref": "pending"},
        "instruction": "재시도에 맞게 본문을 수정한다.", "evidence_ids": ["request"],
    })
    receipt = third["response"].result.actions[0]
    assert receipt.status == "success"
    assert receipt.operation.operation_id != operation.operation_id
    assert receipt.operation.answer_hash == third["response"].result.content_hash
    assert Path(receipt.file_path).read_text(encoding="utf-8-sig") == "보충 내용을 반영한 새 본문"
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert third["runtime"].pending_action is None


@pytest.mark.parametrize("relation", ["correction", "supplement"])
def test_destination_only_followup_keeps_failed_save_identity(tmp_path, monkeypatch, relation):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    conversation = _SaveConversation(["재시도할 본문"], slack=lambda **_kwargs: {"status": "success"})
    with monkeypatch.context() as failure:
        _fail_save_publication(failure)
        first = conversation.turn("설명을 작성하고 저장한 뒤 Slack으로 보내줘", body={
            "kind": "compose", "instruction": "설명을 작성한다.", "evidence_ids": ["request"],
        }, actions={name: {"intent": "requested", "evidence_ids": ["request"]}
                    for name in ("save_text", "slack_notify")})
    operation = first["runtime"].pending_action.save_operation
    manifest_path = next(tmp_path.glob("*.json"))
    reserved_manifest = manifest_path.read_bytes()

    result = conversation.turn("Slack 수신처는 C123이야", relation=relation, actions={},
                               slack_destination={"channel_id": "C123"})
    receipt = result["response"].result.actions[0]
    assert receipt.status == "success"
    assert receipt.operation == operation
    assert result["response"].result.content == first["response"].result.content
    assert manifest_path.read_bytes() == reserved_manifest
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert result["runtime"].pending_action is None


def test_destination_correction_does_not_repeat_completed_save(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    conversation = _SaveConversation(["이미 저장한 본문"], slack=lambda **_kwargs: {"status": "success"})
    first = conversation.turn("설명을 작성하고 저장한 뒤 Slack으로 보내줘", body={
        "kind": "compose", "instruction": "설명을 작성한다.", "evidence_ids": ["request"],
    }, actions={name: {"intent": "requested", "evidence_ids": ["request"]}
                for name in ("save_text", "slack_notify")})
    receipt = first["response"].result.actions[0]
    assert receipt.status == "success"
    path = Path(receipt.file_path)
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    manifest_path = path.with_name(path.name + ".json")
    original_manifest = manifest_path.read_bytes()

    result = conversation.turn("Slack 수신처는 C123이야", relation="correction", actions={},
                               slack_destination={"channel_id": "C123"})
    assert result["runtime"].request_contract.actions.save_text.intent == "not_requested"
    assert [(item.kind, item.status) for item in result["response"].result.actions] == [("slack_notify", "success")]
    assert result["response"].result.content == first["response"].result.content
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime
    assert manifest_path.read_bytes() == original_manifest
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert result["runtime"].pending_action is None


@pytest.mark.parametrize("change", ["format", "reference"])
def test_failed_save_body_changes_in_supplement_create_new_operation(tmp_path, monkeypatch, change):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    conversation = _SaveConversation(["원래 첫 줄\n원래 둘째 줄", "한 줄로 바뀐 본문"])
    with monkeypatch.context() as failure:
        _fail_save_publication(failure)
        first = conversation.turn("설명을 작성하고 저장해줘", body={
            "kind": "compose", "instruction": "설명을 작성한다.", "evidence_ids": ["request"],
        })
    operation = first["runtime"].pending_action.save_operation
    if change == "format":
        result = conversation.turn("그 본문을 한 줄로 고쳐줘", relation="supplement", actions={}, body={
            "kind": "transform_answer", "source": {"ref": "pending"},
            "instruction": "한 줄로 수정한다.", "evidence_ids": ["request"],
        }, answer={"format": [{"kind": "line_count", "mode": "required", "value": 1,
                               "evidence_ids": ["request"]}]})
        expected = "한 줄로 바뀐 본문"
    else:
        expected = "대신 저장할 원문"
        result = conversation.turn(expected, relation="supplement", actions={}, body=lambda turn_id: {
            "kind": "copy_input", "source": {"turn_id": turn_id, "quote": expected},
        })
    receipt = result["response"].result.actions[0]
    assert receipt.status == "success"
    assert receipt.operation.operation_id != operation.operation_id
    assert Path(receipt.file_path).read_text(encoding="utf-8-sig") == expected
    assert result["runtime"].pending_action is None


@pytest.mark.parametrize("change_body", [False, True])
def test_explicit_resave_preserves_existing_file_and_retention(tmp_path, monkeypatch, change_body):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    conversation = _SaveConversation(["보존할 첫 본문", "수정한 둘째 본문"])
    first = conversation.turn("설명을 작성해 저장하고 Slack으로 보내줘", body={
        "kind": "compose", "instruction": "설명을 작성한다.", "evidence_ids": ["request"],
    }, actions={name: {"intent": "requested", "evidence_ids": ["request"]}
                for name in ("save_text", "slack_notify")})
    original = first["response"].result.actions[0]
    assert original.status == "success"
    path = Path(original.file_path)
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    manifest_path = path.with_name(path.name + ".json")
    manifest_bytes = manifest_path.read_bytes()
    body = {"kind": "copy_answer", "source": {"ref": "pending"}}
    if change_body:
        body = {"kind": "transform_answer", "source": {"ref": "pending"},
                "instruction": "둘째 본문으로 수정한다.", "evidence_ids": ["request"]}
    second = conversation.turn("본문을 수정해 별도로 저장해줘" if change_body else "같은 본문을 별도로 다시 저장해줘",
                               relation="correction", body=body)
    receipt = second["response"].result.actions[0]
    assert receipt.status == "success"
    assert receipt.operation.operation_id != original.operation.operation_id
    assert Path(receipt.file_path).read_text(encoding="utf-8-sig") == (
        "수정한 둘째 본문" if change_body else "보존할 첫 본문"
    )
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime
    assert manifest_path.read_bytes() == manifest_bytes
    assert len(list(tmp_path.glob("*.txt"))) == 2
