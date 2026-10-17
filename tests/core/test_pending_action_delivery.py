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
from src.core.answer_schema import AnswerResponse, export_answer_text, text_document
from src.core.contracts import PlannerState
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import RequestContract
from src.infra.settings import AppSettings
from src.infra.tools.save_text import build_save_text_tool
from src.infra.tools.slack_notify import build_slack_notify_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
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


def _manager(graph):
    manager = AgentFlowManager.__new__(AgentFlowManager)
    manager.settings = AppSettings(_env_file=None, openai_api_key="test-key", tavily_api_key="test-key")
    manager.graph = graph
    manager.messages = []
    return manager


def test_separate_task_discards_the_pending_action(delivery_tools):
    """별개의 새 작업은 이전 전송 의사를 다음 턴으로 승계하지 않는다."""
    save, slack, messages, _output = delivery_tools
    manager = _manager(_ConsumerGraph([
        _contract(slack="requested"), _contract(), _contract(),
    ], ["원 본문", "새 작업의 답변", "독립 답변"], save, slack))
    manager.run_agent_flow("이 내용을 Slack으로 보내줘")
    assert manager._ensure_session().pending_action is not None

    second = manager.run_agent_flow("다른 주제를 설명해줘")
    assert export_answer_text(AnswerResponse.model_validate(second["response"])) == "새 작업의 답변"
    assert manager._ensure_session().pending_action is None
    manager.run_agent_flow("C123")

    assert messages == []
