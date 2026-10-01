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
















