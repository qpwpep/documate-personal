import json
from pathlib import Path

import pytest

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.planner_schema import InitialPlannerOutput
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
from tests.core.helpers import make_decisions_client, mock_chat_model, structured_model
from tests.core.test_pending_request_delivery import delivery_tools
from tests.synthesis_fixtures import synthesis_excerpt_limits


def _current_turn(messages):
    context = next(message for message in messages if message.name == "request_context")
    return json.loads(context.content.split("\n", 1)[1])["user_turn_ledger"][-1]


class PlannerBoundary:
    def __init__(self, previous, save_intent):
        self.previous = previous
        self.save_intent = save_intent

    def invoke(self, messages):
        user = _current_turn(messages)
        contract = WireRequestContract.model_validate({
            "slack_recipient": {"state": "omitted"},
            "body": {"kind": "transform_answer", "source": {"ref": "previous"},
                     "instruction": "세 줄로 줄여줘", "evidence_ids": ["format"]},
            "actions": {"save_text": {"intent": self.save_intent, "evidence_ids": ["action"]}},
            "answer": {"format": [{"kind": "line_count", "mode": "required", "value": 3, "evidence_ids": ["format"]}]},
            "evidence": [
                {"id": "format", "turn_id": user["turn_id"], "quote": user["text"], "scope": "answer.format.line_count", "interpretation": "instruction"},
                {"id": "action", "turn_id": user["turn_id"], "quote": user["text"], "scope": "actions.save_text",
                 "interpretation": "instruction" if self.save_intent == "requested" else "negation"},
            ],
        })
        return InitialPlannerOutput(use_retrieval=False, tasks=[], request_contract=contract).model_dump(mode="json")


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
        planner_node=make_planner_node(structured_model(PlannerBoundary(previous, save_intent).invoke, InitialPlannerOutput), False),
        retrieve_dispatch_node=make_retrieve_dispatch_node(unavailable_boundary, unavailable_boundary, False),
        synthesize_node=make_synthesize_node(mock_chat_model(RepairingModelBoundary().invoke)),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client()),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False, decisions_client=make_decisions_client()),
        action_postprocess_node=make_action_postprocess_node(build_save_text_tool(), unavailable_boundary, False),
        memory_policy=policy,
    )
    query = "방금 답변을 세 줄로 줄여 저장해줘" if save_intent == "requested" else "방금 답변을 세 줄로 줄이고 저장하지 마"
    result = graph.invoke(build_graph_state_input(
        user_input=query, previous_response=previous, retry=RetryState(max_retries=2),
    ))
    answer = result["response"].result
    expected = "첫 요약\n둘째 요약\n셋째 요약"
    assert export_answer_text(answer) == expected
    assert result["response"].kind == "answer"
    assert result["response"].synthesis_attempt == 2
    assert result["response"].request_id == result["runtime"].request_contract.request_id
    assert result["response"].contract_revision == result["runtime"].request_contract.revision
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
            user = _current_turn(messages)
            payload = {
                "slack_recipient": {"state": "omitted"},
                "actions": {"save_text": {"intent": "requested", "evidence_ids": ["save"]}},
                "evidence": [{"id": "save", "turn_id": user["turn_id"], "quote": user["text"],
                              "scope": "current_request", "interpretation": "instruction"}],
            }
            if self.pending_request_id is None:
                payload["body"] = {"kind": "compose", "instruction": "설명을 작성한다.", "evidence_ids": ["save"]}
                payload["answer"] = {"sentence_counts": [{
                    "scope": None, "minimum": 1, "maximum": 1, "evidence_ids": ["save"],
                }]}
            else:
                payload.update(relation="supplement", target_request_id=self.pending_request_id,
                               body={"kind": "copy_answer", "source": {"ref": "pending"}})
            return InitialPlannerOutput(use_retrieval=False, tasks=[],
                                 request_contract=WireRequestContract.model_validate(payload)).model_dump(mode="json")

    planner_boundary = DeliveryPlannerBoundary()
    policy = ConversationMemoryPolicy()
    graph = build_graph(
        state_type=GraphState, add_user_node=add_user_message,
        summarize_node=make_summarize_node(None, False, policy=policy),
        planner_node=make_planner_node(structured_model(planner_boundary.invoke, InitialPlannerOutput), False),
        retrieve_dispatch_node=make_retrieve_dispatch_node(unavailable_boundary, unavailable_boundary, False),
        synthesize_node=make_synthesize_node(mock_chat_model(
            lambda _messages: text_document("설명을 작성했습니다.").model_dump(mode="json"),
        )),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client()),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False, decisions_client=make_decisions_client()),
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
            user_input="설명을 한 문장으로 작성하고 파일로 저장해줘", session_id="save-retry-session",
        ))
    failed_receipt = first["response"].result.actions[0]
    pending = first["runtime"].pending_request
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
        previous_response=first["response"].result, pending_request=pending,
    ))
    completed_receipt = second["response"].result.actions[0]
    assert second["runtime"].request_contract.revision == pending.contract.revision + 1
    assert second["runtime"].request_contract.body.kind == "copy_answer"
    assert second["runtime"].request_contract.body.source.ref == "pending"
    assert second["runtime"].request_contract.actions.save_text.intent == "requested"
    assert second["runtime"].request_contract.answer.sentence_counts == pending.contract.answer.sentence_counts
    assert pending.contract.answer.sentence_counts[0].minimum == pending.contract.answer.sentence_counts[0].maximum == 1
    assert second["response"].result.content == first["response"].result.content
    assert completed_receipt.status == "success"
    assert completed_receipt.operation == failed_receipt.operation
    assert completed_receipt.operation.target_kind == "compose"
    assert second["response"].save_operation_binding_sha256 == failed_receipt.operation.binding_sha256
    assert second["runtime"].pending_request is None
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
            planner_node=make_planner_node(structured_model(self.invoke, InitialPlannerOutput), False),
            retrieve_dispatch_node=make_retrieve_dispatch_node(unavailable_boundary, unavailable_boundary, False),
            synthesize_node=make_synthesize_node(mock_chat_model(self._DocumentBoundary(self.documents).invoke)),
            pre_synthesis_validation_node=make_pre_synthesis_validation_node(False, excerpt_limits=synthesis_excerpt_limits(), decisions_client=make_decisions_client()),
            post_synthesis_validation_node=make_post_synthesis_validation_node(False, decisions_client=make_decisions_client()),
            action_postprocess_node=make_action_postprocess_node(build_save_text_tool(), slack, False),
            memory_policy=policy,
        )

    class _DocumentBoundary:
        def __init__(self, documents):
            self.documents = documents

        def invoke(self, _messages):
            return text_document(next(self.documents)).model_dump(mode="json")

    def invoke(self, messages):
        user = _current_turn(messages)
        payload = {
                "slack_recipient": {"state": "omitted"},
            "actions": {"save_text": {"intent": "requested", "evidence_ids": ["request"]}},
            "evidence": [{"id": "request", "turn_id": user["turn_id"], "quote": user["text"],
                          "scope": "current_request", "interpretation": "instruction"}],
            **self.payload,
        }
        if self.result is not None:
            payload["target_request_id"] = self.result["runtime"].pending_request.contract.request_id
        if callable(payload.get("body")):
            payload["body"] = payload["body"](user["turn_id"])
        return InitialPlannerOutput(use_retrieval=False, tasks=[],
                             request_contract=WireRequestContract.model_validate(payload)).model_dump(mode="json")

    def turn(self, query, *, session_metadata=None, **payload):
        self.payload = payload
        previous = self.result
        self.result = self.graph.invoke(build_graph_state_input(
            user_input=query, session_id="save-conversation",
            messages=list(previous["messages"]) if previous else [],
            user_turns=previous["runtime"].user_turns if previous else (),
            previous_response=previous["response"].result if previous else None,
            pending_request=previous["runtime"].pending_request if previous else None,
            session_metadata=session_metadata,
        ))
        return self.result


def _fail_save_publication(monkeypatch):
    real_link = saved_artifacts.os.link

    def fail_payload_publication(source, destination):
        if Path(destination).suffix == ".txt":
            raise OSError("temporary storage failure")
        return real_link(source, destination)

    monkeypatch.setattr(saved_artifacts.os, "link", fail_payload_publication)


def test_destination_correction_does_not_repeat_completed_save(tmp_path, monkeypatch, delivery_tools):
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    _save, slack, messages, _output = delivery_tools
    conversation = _SaveConversation(["이미 저장한 본문"], slack=slack)
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
                               slack_recipient={"state": "explicit", "selector": {"kind": "channel", "value": "C123"}, "evidence_ids": ["request"]})
    assert result["runtime"].request_contract.actions.save_text.intent == "not_requested"
    assert [(item.kind, item.status) for item in result["response"].result.actions] == [("slack_notify", "success")]
    assert result["response"].result.content == first["response"].result.content
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime
    assert manifest_path.read_bytes() == original_manifest
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert result["runtime"].pending_request is None
    assert [item["payload"]["channel"] for item in messages] == ["C123"]


def test_content_supplement_preserves_completed_save_while_preparing_new_body(tmp_path, monkeypatch):
    """Additional content changes the pending answer without reauthorizing a saved file."""
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    conversation = _SaveConversation([
        "시간복잡도는 O(n)이다. 공간복잡도는 O(1)이다.",
        "시간복잡도는 O(n)이다. 공간복잡도는 O(1)이다. 빈 입력이면 즉시 반환한다.",
    ])
    original_conditions = [
        {"kind": "custom", "mode": "required", "description": description,
         "replaces": None, "evidence_ids": ["request"]}
        for description in ("시간복잡도를 설명한다", "공간복잡도를 설명한다")
    ]
    first = conversation.turn(
        "시간복잡도와 공간복잡도를 설명해서 저장하고 Slack으로 보내줘",
        body={"kind": "compose", "instruction": "시간복잡도와 공간복잡도를 설명한다.",
              "evidence_ids": ["request"]},
        answer={"content": original_conditions},
        actions={name: {"intent": "requested", "evidence_ids": ["request"]}
                 for name in ("save_text", "slack_notify")},
    )
    receipt = first["response"].result.actions[0]
    assert receipt.kind == "save_text" and receipt.status == "success"
    pending = first["runtime"].pending_request
    assert pending is not None and pending.completed_actions == ("save_text",)
    path = Path(receipt.file_path)
    manifest_path = path.with_name(path.name + ".json")
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    original_manifest = manifest_path.read_bytes()

    result = conversation.turn(
        "앞의 조건은 유지하고 빈 입력 처리도 설명에 추가해줘",
        relation="supplement", actions={},
        body={"kind": "transform_answer", "source": {"ref": "pending"},
              "instruction": "기존 시간복잡도와 공간복잡도 설명에 빈 입력 처리를 추가한다.",
              "evidence_ids": ["request"]},
        answer={"content": [
            {"kind": "custom", "mode": "required", "description": "빈 입력 처리를 설명한다",
             "replaces": None, "evidence_ids": ["request"]},
        ]},
    )
    contract = result["runtime"].request_contract
    assert [item.description for item in contract.answer.content] == [
        "시간복잡도를 설명한다", "공간복잡도를 설명한다", "빈 입력 처리를 설명한다",
    ]
    assert contract.actions.save_text.intent == "not_requested"
    assert export_answer_text(result["response"].result) == (
        "시간복잡도는 O(n)이다. 공간복잡도는 O(1)이다. 빈 입력이면 즉시 반환한다."
    )
    assert all(item.kind != "save_text" for item in result["response"].result.actions)
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime
    assert manifest_path.read_bytes() == original_manifest
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1
    pending = result["runtime"].pending_request
    assert pending is not None and pending.body_prepared
    assert pending.contract.actions.slack_notify.intent == "requested"


def test_content_reconfirmation_keeps_failed_save_operation_frozen(tmp_path, monkeypatch):
    """A new evidence binding for the same condition does not replace a failed write."""
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    conversation = _SaveConversation(["시간복잡도는 O(n)이다."])
    condition = {"kind": "custom", "mode": "required", "description": "시간복잡도를 설명한다",
                 "replaces": None, "evidence_ids": ["request"]}
    with monkeypatch.context() as failure:
        _fail_save_publication(failure)
        first = conversation.turn(
            "시간복잡도를 설명하고 저장해줘",
            body={"kind": "compose", "instruction": "시간복잡도를 설명한다.",
                  "evidence_ids": ["request"]},
            answer={"content": [condition]},
        )
    failed_receipt = first["response"].result.actions[0]
    assert failed_receipt.status == "error" and failed_receipt.error_code == "write_failed"
    pending = first["runtime"].pending_request
    assert pending is not None and pending.body_prepared
    assert pending.save_operation == failed_receipt.operation
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert list(tmp_path.glob("*.txt")) == []

    result = conversation.turn(
        "시간복잡도 설명 조건은 그대로야. 그 본문 그대로 저장을 다시 시도해줘",
        relation="supplement", body={"kind": "copy_answer", "source": {"ref": "pending"}},
        answer={"content": [condition]},
    )
    contract = result["runtime"].request_contract
    assert contract.revision == pending.contract.revision + 1
    assert len(contract.answer.content) == 1
    assert contract.answer.content[0].description == "시간복잡도를 설명한다"
    assert set(contract.answer.content[0].evidence_ids) > set(pending.contract.answer.content[0].evidence_ids)
    assert result["response"].result.content_hash == first["response"].result.content_hash
    receipt = result["response"].result.actions[0]
    assert receipt.status == "success"
    assert receipt.operation == failed_receipt.operation
    assert receipt.operation.target_kind == "compose"
    assert result["response"].save_operation_binding_sha256 == failed_receipt.operation.binding_sha256
    assert result["runtime"].pending_request is None
    assert Path(receipt.file_path).read_bytes() == "시간복잡도는 O(n)이다.".encode("utf-8-sig")
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1
