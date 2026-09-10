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














