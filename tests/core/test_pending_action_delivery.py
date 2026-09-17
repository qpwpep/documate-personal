from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from langchain_core.messages import HumanMessage
from slack_sdk.web import WebClient

from src.app.agent_manager import AgentFlowManager
from src.core.answer_schema import AnswerResponse, export_answer_text, finalize_answer, text_document
from src.core.contracts import PlannerState
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import RequestContract
from src.infra.settings import AppSettings
from src.infra.tools.save_text import build_save_text_tool
from src.infra.tools.slack_notify import build_slack_notify_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node


def _contract(*, save="not_requested", slack="not_requested", **updates):
    return RequestContract.model_validate({
        "actions": {
            "save_text": {"intent": save, "evidence_ids": ["request"]},
            "slack_notify": {"intent": slack, "evidence_ids": ["request"]},
        },
        "evidence": [{"id": "request", "turn_id": "current", "quote": "사용자 요청", "scope": "current_request", "interpretation": "instruction"}],
        **updates,
    })


@pytest.fixture
def delivery_tools(tmp_path: Path, monkeypatch, request):
    """실제 저장 도구와 Slack SDK를 임시 파일 및 로컬 HTTP 경계에 연결한다."""
    messages = []
    responses = iter(getattr(request, "param", []))

    class SlackHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            messages.append({"path": self.path, "payload": json.loads(body)})
            payload = json.dumps(next(responses, {"ok": True, "channel": "C123", "ts": "1.0"})).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlackHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    client = WebClient(token="test-token", base_url=f"http://127.0.0.1:{server.server_port}/", retry_handlers=[])
    monkeypatch.setattr("src.infra.tools.slack_notify.create_slack_client", lambda _token: client)
    output = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    settings = AppSettings(_env_file=None, openai_api_key="test-key", tavily_api_key="test-key", slack_bot_token="test-token")
    try:
        yield build_save_text_tool(), build_slack_notify_tool(settings), messages, output
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


class _DocumentModelBoundary:
    def __init__(self, documents):
        self.documents = iter(documents)

    def invoke(self, _messages):
        return text_document(next(self.documents)).model_dump(mode="json")


class _ConsumerGraph:
    """정답 계약 fixture를 받아 실제 합성과 액션 소비 경로를 실행한다."""

    def __init__(self, contracts, documents, save, slack):
        self.contracts = iter(contracts)
        self.synthesize = make_synthesize_node(_DocumentModelBoundary(documents))
        self.validate = make_post_synthesis_validation_node(verbose=False)
        self.actions = make_action_postprocess_node(save, slack, False)

    def invoke(self, state):
        state = dict(state)
        contract = next(self.contracts)
        if callable(contract):
            contract = contract(state["runtime"])
        state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
        state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[], request_contract=contract.to_wire()))
        state["messages"] = [*state["messages"], HumanMessage(content=state["runtime"].user_input)]
        for node in (self.synthesize, self.validate, self.actions):
            updates = node(state)
            messages = [*state["messages"], *updates.pop("messages", [])]
            state.update(updates)
            state["messages"] = messages
        return state


def _manager(graph):
    manager = AgentFlowManager.__new__(AgentFlowManager)
    manager.settings = AppSettings(_env_file=None, openai_api_key="test-key", tavily_api_key="test-key")
    manager.graph = graph
    manager.messages = []
    return manager


def _resume(runtime):
    pending = runtime.pending_action
    assert pending is not None
    payload = pending.contract.model_dump(mode="json")
    payload.update({
        "relation": "supplement", "target_request_id": pending.contract.request_id,
        "revision": pending.contract.revision + 1,
        "body": {"kind": "copy_answer", "source": {"ref": "pending", "response_hash": pending.response.content_hash}},
        "slack_destination": {"channel_id": "C123"},
        "missing_info": [], "body_request": None,
    })
    return RequestContract.model_validate(payload)


def _clarify(runtime):
    payload = _resume(runtime).model_dump(mode="json")
    payload.update({"slack_destination": None, "missing_info": [
        {"slot": "slack_destination", "reason": "unknown_id", "question": "팀의 channel_id를 알려주세요."},
    ]})
    return RequestContract.model_validate(payload)


def test_transform_then_save_writes_the_modified_three_line_body(delivery_tools):
    """이전 답변 수정 계약은 세 줄의 새 본문을 실제 파일에 저장한다."""
    save, slack, messages, output = delivery_tools
    previous = finalize_answer(text_document("첫 줄\n둘째 줄\n셋째 줄\n넷째 줄"), [])
    transformed = "첫 요약\n둘째 요약\n셋째 요약"
    contract = _contract(
        save="requested",
        body={
            "kind": "transform_answer", "source": {"ref": "previous", "response_hash": previous.content_hash},
            "instruction": "세 줄로 줄여줘", "evidence_ids": ["request"],
        },
        answer={"format": [{"kind": "line_count", "mode": "required", "value": 3, "evidence_ids": ["request"]}]},
    )
    manager = _manager(_ConsumerGraph([contract], [transformed], save, slack))
    manager._ensure_session().previous_response = previous

    result = manager.run_agent_flow("방금 답변을 세 줄로 줄여 저장해줘")

    answer = AnswerResponse.model_validate(result["response"])
    saved_files = list(output.glob("*.txt"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text(encoding="utf-8-sig") == transformed == export_answer_text(answer)
    assert len(export_answer_text(answer).splitlines()) == 3
    assert answer.content != previous.content
    assert messages == []


def test_destination_followup_delivers_the_saved_body_once(delivery_tools):
    """목적지 보충은 보류한 원 본문을 전송하며 먼저 성공한 저장을 반복하지 않는다."""
    save, slack, messages, output = delivery_tools
    original = "전송할 원래 답변"
    snapshot = build_snapshot(
        source_uri="https://docs.example.com/pending", title="보존할 출처", media_type="text/plain",
        source_type="official", content=original, parser="test", parser_version="1",
    )
    evidence = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="p1", kind="paragraph", text=original))
    previous = finalize_answer(text_document(original, basis="source", refs=[evidence.id]), [evidence])
    manager = _manager(_ConsumerGraph([
        _contract(save="requested", slack="requested", body={
            "kind": "copy_answer", "source": {"ref": "previous", "response_hash": previous.content_hash},
        }), _clarify, _resume,
    ], [], save, slack))
    manager._ensure_session().previous_response = previous

    first = AnswerResponse.model_validate(manager.run_agent_flow("설명하고 파일로 저장한 뒤 Slack으로 보내줘")["response"])
    pending = manager._ensure_session().pending_action
    assert pending is not None
    assert pending.completed_actions == ("save_text",)
    saved_file = Path(first.actions[0].file_path)
    saved_bytes = saved_file.read_bytes()
    saved_time = saved_file.stat().st_mtime_ns
    assert first.content == pending.response.content
    assert "알려주세요" in first.actions[1].message
    assert messages == []

    clarification = AnswerResponse.model_validate(manager.run_agent_flow("우리 팀에")["response"])
    assert clarification.content == first.content
    assert clarification.actions[0] == first.actions[0]
    assert clarification.actions[1].message == "팀의 channel_id를 알려주세요."
    assert manager._ensure_session().previous_response.content == first.content
    assert manager._ensure_session().pending_action.response.content == first.content

    second = AnswerResponse.model_validate(manager.run_agent_flow("C123")["response"])

    delivered = export_answer_text(previous, include_sources=True)
    assert messages == [{"path": "/chat.postMessage", "payload": {"channel": "C123", "text": delivered}}]
    assert saved_file.read_text(encoding="utf-8-sig") == delivered
    assert second.content == previous.content
    assert second.citations == previous.citations
    assert saved_file.read_bytes() == saved_bytes
    assert saved_file.stat().st_mtime_ns == saved_time
    assert len(list(output.glob("*.txt"))) == 1
    assert manager._ensure_session().pending_action is None
    assert second.actions[0] == first.actions[0]
    assert [(receipt.kind, receipt.status) for receipt in second.actions] == [
        ("save_text", "success"), ("slack_notify", "success"),
    ]


@pytest.mark.parametrize("relation", ["cancel", "new"])
def test_cancel_or_separate_task_discards_the_pending_action(delivery_tools, relation):
    """취소와 별개의 새 작업은 이전 전송 의사를 다음 턴으로 승계하지 않는다."""
    save, slack, messages, _output = delivery_tools
    def next_request(runtime):
        return _contract(relation=relation,
                         target_request_id=runtime.pending_action.contract.request_id if relation == "cancel" else None)
    manager = _manager(_ConsumerGraph([
        _contract(slack="requested"), next_request, _contract(),
    ], ["원 본문", "새 작업의 답변", "독립 답변"], save, slack))
    manager.run_agent_flow("이 내용을 Slack으로 보내줘")
    assert manager._ensure_session().pending_action is not None

    manager.run_agent_flow("취소해" if relation == "cancel" else "다른 주제를 설명해줘")
    assert manager._ensure_session().pending_action is None
    manager.run_agent_flow("C123")

    assert messages == []


def test_graph_failure_cannot_mutate_the_session_pending_body(delivery_tools):
    """실패한 graph가 입력을 변경해도 세션이 보관한 전달 본문과 이전 답변은 보존된다."""
    save, slack, messages, _output = delivery_tools
    manager = _manager(_ConsumerGraph([_contract(slack="requested")], ["보존할 본문"], save, slack))
    manager.run_agent_flow("Slack으로 보내줘")
    session = manager._ensure_session()
    original_pending = session.pending_action.model_dump(mode="json")
    original_previous = session.previous_response.model_dump(mode="json")

    class MutatingGraph:
        def invoke(self, state):
            state["runtime"].pending_action.response.content.blocks[0].content[0].text = "실패 중 변조"
            state["runtime"].previous_response.content.blocks[0].content[0].text = "실패 중 변조"
            raise RuntimeError("failed graph")

    manager.graph = MutatingGraph()
    result = manager.run_agent_flow("대상을 다시 확인해줘")

    assert result["debug"]["observability_status"] == "failed"
    assert session.pending_action.model_dump(mode="json") == original_pending
    assert session.previous_response.model_dump(mode="json") == original_previous
    assert messages == []


def test_forbidding_the_remaining_pending_action_clears_it(delivery_tools):
    """보류 전송이 명시적으로 금지되면 원 본문을 보존한 채 pending을 종료한다."""
    save, slack, messages, _output = delivery_tools

    def forbid(runtime):
        payload = _resume(runtime).model_dump(mode="json")
        payload["actions"]["slack_notify"]["intent"] = "forbidden"
        return RequestContract.model_validate(payload)

    manager = _manager(_ConsumerGraph([_contract(slack="requested"), forbid], ["보존할 원 본문"], save, slack))
    first = AnswerResponse.model_validate(manager.run_agent_flow("이 답변을 Slack으로 보내줘")["response"])
    assert manager._ensure_session().pending_action is not None

    second = AnswerResponse.model_validate(manager.run_agent_flow("C123이지만 보내지 마")["response"])

    assert second.content == first.content
    assert manager._ensure_session().pending_action is None
    assert messages == []


def test_uncertain_cancellation_preserves_pending_until_clarified(delivery_tools):
    """취소 의사가 불명확한 보충 질문은 원 본문과 보류 요청을 폐기하지 않는다."""
    save, slack, messages, _output = delivery_tools
    def uncertain_cancel(runtime):
        pending = runtime.pending_action
        return _contract(relation="cancel", target_request_id=pending.contract.request_id,
                         request_id=pending.contract.request_id, revision=pending.contract.revision + 1,
                         missing_info=[{"slot": "slack_intent", "reason": "unclear", "question": "전송을 취소할까요?"}])
    manager = _manager(_ConsumerGraph([
        _contract(slack="requested"),
        uncertain_cancel,
    ], ["보존할 원 본문"], save, slack))
    manager.run_agent_flow("이 답변을 Slack으로 보내줘")
    pending = manager._ensure_session().pending_action

    clarification = AnswerResponse.model_validate(manager.run_agent_flow("취소할까?")["response"])

    assert export_answer_text(clarification) == "전송을 취소할까요?"
    retained = manager._ensure_session().pending_action
    assert retained.response == pending.response
    assert retained.completed_actions == pending.completed_actions
    assert retained.body_prepared == pending.body_prepared
    assert retained.phase == "awaiting_input"
    assert retained.contract.request_id == pending.contract.request_id
    assert retained.contract.revision == pending.contract.revision + 1
    assert any(item.slot == "slack_intent" for item in retained.contract.missing_info)
    assert messages == []


@pytest.mark.parametrize("cancel_intent", ["unresolved", "not_requested"])
def test_uncertain_cancellation_cannot_be_resolved_by_a_destination_only_supplement(delivery_tools, cancel_intent):
    """취소 의사의 모호함은 실제 같은 요청의 최신 계약에 남아 목적지 보충만으로 전송이 되살아나지 않는다."""
    from src.core.contracts.boundary.graph import build_graph_state_input
    from src.core.contracts.graph_state import PendingAction
    from src.core.request_contracts import UserTurnSnapshot, WireRequestContract
    from src.runtime.nodes.planner import make_planner_node

    save, slack, delivered, output = delivery_tools
    original_response = finalize_answer(text_document("취소 여부를 확인 중인 원본문"), [])
    original_contract = _contract(save="requested", slack="requested")
    original = PendingAction(contract=original_contract, response=original_response,
                             completed_actions=("save_text",), body_prepared=True)
    actions = make_action_postprocess_node(save, slack, False)

    def run_turn(query, turn_id, proposal, pending):
        class PlannerBoundary:
            def invoke(self, _messages):
                return PlannerOutput(use_retrieval=False, tasks=[], request_contract=proposal)

        state = build_graph_state_input(
            user_input=query, current_turn_id=turn_id,
            user_turns=(UserTurnSnapshot(turn_id=turn_id, text=query),),
            messages=[HumanMessage(content=query, id=turn_id)],
            previous_response=original_response, pending_action=pending,
        )
        for node in (make_planner_node(PlannerBoundary(), False),
                     make_synthesize_node(_DocumentModelBoundary([])),
                     make_post_synthesis_validation_node(False), actions):
            state.update(node(state))
        return state

    cancellation = WireRequestContract.model_validate({
        "relation": "cancel", "target_request_id": original_contract.request_id,
        "actions": {"slack_notify": {"intent": cancel_intent, "evidence_ids": ["cancel"]}},
        "missing_info": [{"slot": "slack_intent", "reason": "unclear", "question": "전송을 취소할까요?"}],
        "evidence": [{"id": "cancel", "turn_id": "cancel-turn", "quote": "취소할까?",
                      "scope": "actions.slack_notify", "interpretation": "reference"}],
    })
    first = run_turn("취소할까?", "cancel-turn", cancellation, original)
    retained = first["runtime"].pending_action
    destination = WireRequestContract.model_validate({
        "relation": "supplement", "target_request_id": original_contract.request_id,
        "slack_destination": {"channel_id": "C123"},
        "evidence": [{"id": "destination", "turn_id": "destination-turn", "quote": "C123",
                      "scope": "slack_destination", "interpretation": "reference"}],
    })

    second = run_turn("C123", "destination-turn", destination, retained)

    assert delivered == []
    assert not output.exists()
    assert retained.response == original_response
    assert retained.completed_actions == ("save_text",)
    assert retained.body_prepared
    assert retained.phase == "awaiting_input"
    assert retained.contract.request_id == original_contract.request_id
    assert retained.contract.revision == original_contract.revision + 1
    assert any(item.slot == "slack_intent" for item in retained.contract.missing_info)
    assert second["runtime"].pending_action is not None
    assert any(item.slot == "slack_intent" for item in second["runtime"].pending_action.contract.missing_info)
