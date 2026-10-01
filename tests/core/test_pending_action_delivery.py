from __future__ import annotations

import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import pytest
from langchain_core.messages import HumanMessage
from slack_sdk.web import WebClient

from src.app.agent_manager import AgentFlowManager
from src.core.answer_schema import AnswerResponse, export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, PlannerState
from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import RequestContract, WireRequestContract
from src.core.slack_contract import RecipientSelector, SlackDefault
from src.infra.settings import AppSettings
from src.infra.tools.save_text import build_save_text_tool
from src.infra.tools.slack_notify import build_slack_notify_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.make_graph import build_graph
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.session import add_user_message, make_summarize_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.synthesis_fixtures import synthesis_excerpt_limits


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
    configured_responses = getattr(request, "param", {})
    responses = iter(configured_responses) if isinstance(configured_responses, list) else None
    endpoint_responses = {
        path: iter(items) if isinstance(items, list) else iter([items])
        for path, items in configured_responses.items()
    } if isinstance(configured_responses, dict) else {}

    class SlackHandler(BaseHTTPRequestHandler):
        def _respond(self):
            parsed = urlsplit(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if body and "application/json" in self.headers.get("Content-Type", ""):
                payload = json.loads(body)
            else:
                values = parse_qs(body.decode() if body else parsed.query)
                payload = {key: items[0] for key, items in values.items()}
            messages.append({"path": parsed.path, "payload": payload})
            default = {
                "/users.lookupByEmail": {"ok": True, "user": {"id": "UINTENDED"}},
                "/conversations.open": {"ok": True, "channel": {"id": "D" + payload.get("users", "UTARGET")[1:]}},
                "/chat.postMessage": {"ok": True, "channel": payload.get("channel"), "ts": "1.0"},
            }[parsed.path]
            iterator = endpoint_responses.get(parsed.path, responses)
            response = next(iterator, default) if iterator is not None else default
            if response is None:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_POST = _respond
        do_GET = _respond

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
        yield build_save_text_tool(), build_slack_notify_tool(settings.slack_bot_token), messages, output
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

    def __init__(self, contracts, documents, save, slack, default_slack_recipient=None):
        self.contracts = iter(contracts)
        self.synthesize = make_synthesize_node(_DocumentModelBoundary(documents), excerpt_limits=synthesis_excerpt_limits())
        self.validate = make_post_synthesis_validation_node(verbose=False)
        self.actions = make_action_postprocess_node(save, slack, False, default_slack_recipient)

    def stream(self, state, *, stream_mode):
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
        yield state


class _PlannedConsumerGraph:
    """Supply model proposals while keeping reconciliation and execution real."""

    def __init__(self, proposals, documents, save, slack, default_slack_recipient=None):
        self.proposals = iter(proposals)
        self.proposal = None
        policy = ConversationMemoryPolicy()
        self.graph = build_graph(
            state_type=GraphState, add_user_node=add_user_message,
            summarize_node=make_summarize_node(None, False, policy=policy),
            planner_node=make_planner_node(self._PlannerBoundary(self), False),
            retrieve_dispatch_node=make_retrieve_dispatch_node(self._unavailable_search, self._unavailable_search, False),
            synthesize_node=make_synthesize_node(_DocumentModelBoundary(documents), excerpt_limits=synthesis_excerpt_limits()),
            pre_synthesis_validation_node=make_pre_synthesis_validation_node(False),
            post_synthesis_validation_node=make_post_synthesis_validation_node(False),
            action_postprocess_node=make_action_postprocess_node(save, slack, False, default_slack_recipient),
            memory_policy=policy,
        )

    @staticmethod
    def _unavailable_search(*_args, **_kwargs):
        raise AssertionError("These delivery requests do not use external search.")

    class _PlannerBoundary:
        def __init__(self, owner):
            self.owner = owner

        def invoke(self, messages):
            user = [message for message in messages if isinstance(message, HumanMessage)][-1]
            proposal = WireRequestContract.model_validate({
                "slack_recipient": {"state": "omitted"},
                "evidence": [{"id": "request", "turn_id": user.id, "quote": str(user.content),
                              "scope": "current_request", "interpretation": "instruction"}],
                **self.owner.proposal,
            })
            return PlannerOutput(use_retrieval=False, tasks=[], request_contract=proposal)

    def stream(self, state, *, stream_mode):
        proposal = next(self.proposals)
        self.proposal = proposal(state["runtime"]) if callable(proposal) else proposal
        yield from self.graph.stream(state, stream_mode=stream_mode)


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
        "slack_recipient": _explicit("channel", "C123"),
        "missing_info": [], "body_request": None,
    })
    return RequestContract.model_validate(payload)


def _explicit(kind, value, evidence_id="request"):
    return {"state": "explicit", "selector": {"kind": kind, "value": value}, "evidence_ids": [evidence_id]}


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
    def recipient_followup(runtime, recipient):
        return {
            "relation": "supplement", "target_request_id": runtime.pending_action.contract.request_id,
            "body": {"kind": "copy_answer", "source": {"ref": "pending"}},
            "slack_recipient": recipient,
        }

    manager = _manager(_PlannedConsumerGraph([
        {
            "actions": {name: {"intent": "requested", "evidence_ids": ["request"]}
                        for name in ("save_text", "slack_notify")},
            "body": {"kind": "copy_answer", "source": {"ref": "previous"}},
        },
        lambda runtime: recipient_followup(runtime, {
            "state": "unresolved", "raw_input": "우리 팀", "reason": "ambiguous", "evidence_ids": ["request"],
        }),
        lambda runtime: recipient_followup(runtime, _explicit("channel", "C123")),
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
    assert first.actions[1].slack.failure.code == "recipient_missing"
    assert messages == []

    clarification = AnswerResponse.model_validate(manager.run_agent_flow("우리 팀에")["response"])
    assert clarification.content == first.content
    assert manager._ensure_session().pending_action.save_receipt == first.actions[0]
    assert clarification.actions[0].slack.failure.code == "recipient_ambiguous"
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
    assert [(receipt.kind, receipt.status) for receipt in second.actions] == [
        ("slack_notify", "success"),
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
        def stream(self, state, *, stream_mode):
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
                     make_synthesize_node(_DocumentModelBoundary([]), excerpt_limits=synthesis_excerpt_limits()),
                     make_post_synthesis_validation_node(False), actions):
            state.update(node(state))
        return state

    cancellation = WireRequestContract.model_validate({
        "relation": "cancel", "target_request_id": original_contract.request_id,
        "slack_recipient": {"state": "omitted"},
        "actions": {"slack_notify": {"intent": cancel_intent, "evidence_ids": ["cancel"]}},
        "missing_info": [{"slot": "slack_intent", "reason": "unclear", "question": "전송을 취소할까요?"}],
        "evidence": [{"id": "cancel", "turn_id": "cancel-turn", "quote": "취소할까?",
                      "scope": "actions.slack_notify", "interpretation": "reference"}],
    })
    first = run_turn("취소할까?", "cancel-turn", cancellation, original)
    retained = first["runtime"].pending_action
    destination = WireRequestContract.model_validate({
        "relation": "supplement", "target_request_id": original_contract.request_id,
        "slack_recipient": _explicit("channel", "C123", "destination"),
        "evidence": [{"id": "destination", "turn_id": "destination-turn", "quote": "C123",
                      "scope": "slack_recipient", "interpretation": "reference"}],
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


def _retry(runtime):
    payload = _resume(runtime).model_dump(mode="json")
    payload["slack_recipient"] = runtime.pending_action.contract.slack_recipient.model_dump(mode="json")
    return RequestContract.model_validate(payload)


def _default(user_id="UDEFAULT"):
    return SlackDefault(selector=RecipientSelector(kind="user", value=user_id))


def _slack_action(response):
    return next(receipt for receipt in response.actions if receipt.kind == "slack_notify")


def _message_requests(requests):
    return [request for request in requests if request["path"] == "/chat.postMessage"]


@pytest.mark.parametrize("delivery_tools,error_code", [
    ({"/users.lookupByEmail": {"ok": False, "error": "users_not_found"}}, "target_not_found"),
    ({"/users.lookupByEmail": {"ok": False, "error": "missing_scope"}}, "permission_denied"),
    ({"/users.lookupByEmail": {"ok": False, "error": "ratelimited"}}, "rate_limited"),
], indirect=["delivery_tools"])
def test_explicit_lookup_failure_with_default_has_zero_sends_and_reports_the_reason(delivery_tools, error_code):
    """The manager's public result preserves the explicit identity and Slack failure."""
    save, slack, requests, _output = delivery_tools
    manager = _manager(_PlannedConsumerGraph([
        {"actions": {"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}},
         "slack_recipient": _explicit("email", "intended@example.invalid")},
    ], ["전송할 본문"], save, slack, _default()))

    result = manager.run_agent_flow("intended@example.invalid에 Slack으로 보내줘")
    answer = AnswerResponse.model_validate(result["response"])
    receipt = _slack_action(answer)

    assert _message_requests(requests) == []
    assert receipt.status == "error"
    assert receipt.slack.status == "not_sent"
    assert receipt.slack.intent.selector.value == "intended@example.invalid"
    assert receipt.slack.selection.source == "user_text"
    assert receipt.slack.selection.selector.value == "intended@example.invalid"
    assert receipt.slack.failure.code == error_code
    assert receipt.error == receipt.slack.failure.message
    assert receipt.error_code == f"SLACK_{error_code.upper()}"
    assert manager._ensure_session().pending_action.slack_delivery == receipt.slack
    assert export_answer_text(answer) == "전송할 본문"


@pytest.mark.parametrize("kind,value,expected_channel", [
    ("channel", "C123", "C123"),
    ("user", "UINTENDED", "DINTENDED"),
    ("email", "intended@example.invalid", "DINTENDED"),
])
def test_explicit_success_uses_only_the_requested_recipient(delivery_tools, kind, value, expected_channel):
    save, slack, requests, _output = delivery_tools
    manager = _manager(_ConsumerGraph([
        _contract(slack="requested", slack_recipient=_explicit(kind, value)),
    ], ["선택한 대상으로 전달"], save, slack, _default()))

    answer = AnswerResponse.model_validate(manager.run_agent_flow(f"{value}에 보내줘")["response"])

    assert _message_requests(requests) == [{"path": "/chat.postMessage", "payload": {
        "channel": expected_channel, "text": "선택한 대상으로 전달",
    }}]
    receipt = _slack_action(answer)
    assert receipt.status == "success"
    assert receipt.slack.selection.selector == RecipientSelector(kind=kind, value=value)
    assert receipt.slack.target.channel_id == expected_channel
    assert receipt.slack.message_ts == "1.0"
    assert manager._ensure_session().pending_action is None


def test_omitted_recipient_uses_one_configured_default_and_reports_its_source(delivery_tools):
    save, slack, requests, _output = delivery_tools
    manager = _manager(_ConsumerGraph([_contract(slack="requested")], ["기본 수신자 본문"], save, slack, _default()))

    answer = AnswerResponse.model_validate(manager.run_agent_flow("Slack으로 보내줘")["response"])

    assert len(_message_requests(requests)) == 1
    assert _message_requests(requests)[0]["payload"]["channel"] == "DDEFAULT"
    receipt = _slack_action(answer)
    assert receipt.status == "success"
    assert receipt.slack.intent.state == "omitted"
    assert receipt.slack.selection.source == "configured_default"
    assert receipt.slack.target.user_id == "UDEFAULT"


@pytest.mark.parametrize("recipient", [
    {"state": "unresolved", "raw_input": "우리 팀", "reason": "ambiguous", "evidence_ids": ["request"]},
    {"state": "unresolved", "raw_input": "unknown", "reason": "invalid", "evidence_ids": ["request"]},
])
def test_unresolved_recipient_does_not_use_the_default(delivery_tools, recipient):
    save, slack, requests, _output = delivery_tools
    manager = _manager(_ConsumerGraph([
        _contract(slack="requested", slack_recipient=recipient),
    ], ["전송 보류 본문"], save, slack, _default()))

    answer = AnswerResponse.model_validate(manager.run_agent_flow("우리 팀에 보내줘")["response"])

    assert requests == []
    receipt = _slack_action(answer)
    assert receipt.status == "skipped"
    assert receipt.slack.intent.raw_input == recipient["raw_input"]
    assert receipt.slack.selection is None
    assert receipt.slack.failure.next_action == "correct_input"


@pytest.mark.parametrize("delivery_tools", [{
    "/chat.postMessage": [{"ok": False, "error": "missing_scope"}],
}], indirect=True)
@pytest.mark.parametrize("source", ["explicit", "default"])
def test_pending_retry_ignores_changed_defaults_and_request_metadata(delivery_tools, source):
    save, slack, requests, _output = delivery_tools
    recipient = _explicit("email", "intended@example.invalid") if source == "explicit" else {"state": "omitted"}
    graph = _ConsumerGraph([
        _contract(slack="requested", slack_recipient=recipient), _retry,
    ], ["재시도할 원본문"], save, slack, _default())
    manager = _manager(graph)
    first = AnswerResponse.model_validate(manager.run_agent_flow("Slack으로 보내줘")["response"])
    original = _slack_action(first).slack
    assert original.status == "not_sent"

    graph.actions = make_action_postprocess_node(save, slack, False, _default("UCHANGED"))
    manager.session_metadata = {"slack_recipient": {"kind": "user", "value": "UMETADATA"}}
    second = AnswerResponse.model_validate(manager.run_agent_flow("같은 수신자로 다시 보내줘")["response"])

    delivered = _slack_action(second).slack
    assert delivered.status == "sent"
    assert delivered.selection == original.selection
    assert delivered.target == original.target
    assert second.content == first.content
    assert len(_message_requests(requests)) == 2
    assert {request["payload"]["channel"] for request in _message_requests(requests)} == {original.target.channel_id}
    assert manager._ensure_session().pending_action is None


@pytest.mark.parametrize("delivery_tools", [{
    "/users.lookupByEmail": [
        {"ok": True, "user": {"id": "UINTENDED"}}, {"ok": True, "user": {"id": "UOTHER"}},
    ],
    "/conversations.open": [{"ok": False, "error": "missing_scope"}],
}], indirect=True)
def test_pending_retry_retains_email_to_user_binding_after_open_failure(delivery_tools):
    save, slack, requests, _output = delivery_tools
    manager = _manager(_ConsumerGraph([
        _contract(slack="requested", slack_recipient=_explicit("email", "intended@example.invalid")), _retry,
    ], ["이메일 대상 보존"], save, slack, _default()))

    first = AnswerResponse.model_validate(manager.run_agent_flow("이메일로 보내줘")["response"])
    assert _slack_action(first).slack.resolved_user_id == "UINTENDED"
    assert _message_requests(requests) == []

    second = AnswerResponse.model_validate(manager.run_agent_flow("권한 수정했으니 다시 보내줘")["response"])

    assert _slack_action(second).slack.target.user_id == "UINTENDED"
    assert _message_requests(requests) == [{"path": "/chat.postMessage", "payload": {
        "channel": "DINTENDED", "text": "이메일 대상 보존",
    }}]


@pytest.mark.parametrize("delivery_tools", [{
    "/users.lookupByEmail": {"ok": False, "error": "users_not_found"},
}], indirect=True)
def test_explicit_recipient_correction_preserves_saved_body_and_partial_success(delivery_tools):
    save, slack, requests, output = delivery_tools

    def correct(runtime):
        payload = _resume(runtime).model_dump(mode="json")
        payload["relation"] = "correction"
        payload["slack_recipient"] = _explicit("channel", "CNEW")
        return RequestContract.model_validate(payload)

    manager = _manager(_ConsumerGraph([
        _contract(save="requested", slack="requested", slack_recipient=_explicit("email", "wrong@example.invalid")),
        correct,
    ], ["저장한 원래 본문"], save, slack, _default()))

    first = AnswerResponse.model_validate(manager.run_agent_flow("저장하고 wrong@example.invalid로 보내줘")["response"])
    saved = next(receipt for receipt in first.actions if receipt.kind == "save_text")
    saved_file = Path(saved.file_path)
    saved_bytes, saved_mtime = saved_file.read_bytes(), saved_file.stat().st_mtime_ns
    assert _slack_action(first).slack.status == "not_sent"
    assert _message_requests(requests) == []

    second = AnswerResponse.model_validate(manager.run_agent_flow("수신자를 CNEW로 바꿔서 보내줘")["response"])

    assert _slack_action(second).status == "success"
    assert _slack_action(second).slack.selection.selector.value == "CNEW"
    assert _message_requests(requests) == [{"path": "/chat.postMessage", "payload": {
        "channel": "CNEW", "text": "저장한 원래 본문",
    }}]
    assert second.content == first.content
    assert next(receipt for receipt in second.actions if receipt.kind == "save_text") == saved
    assert saved_file.read_bytes() == saved_bytes
    assert saved_file.stat().st_mtime_ns == saved_mtime
    assert len(list(output.glob("*.txt"))) == 1


@pytest.mark.parametrize("delivery_tools", [{"/chat.postMessage": None}], indirect=True)
def test_unknown_delivery_is_reported_and_pending_retry_does_not_resend(delivery_tools):
    save, slack, requests, _output = delivery_tools
    manager = _manager(_ConsumerGraph([
        _contract(slack="requested", slack_recipient=_explicit("channel", "C123")), _retry,
    ], ["확인할 본문"], save, slack, _default()))

    first = AnswerResponse.model_validate(manager.run_agent_flow("C123으로 보내줘")["response"])
    second = AnswerResponse.model_validate(manager.run_agent_flow("다시 보내줘")["response"])

    receipt = _slack_action(first)
    assert receipt.status == "unknown"
    assert receipt.slack.status == "unknown"
    assert receipt.slack.failure.next_action == "verify_delivery"
    assert _slack_action(second) == receipt
    assert len(_message_requests(requests)) == 1
