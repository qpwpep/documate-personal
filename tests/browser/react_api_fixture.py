"""Serve the real FastAPI and graph with isolated files and fake provider HTTP.

Only model, embedding and Slack service boundaries are replaced. Production
routes, sessions, upload storage, agent runner, exports and SSE encoding are real.
"""
from __future__ import annotations

import argparse
import json
import socket
from contextlib import contextmanager
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import uvicorn
from fastapi import Request
from langchain_core.embeddings import Embeddings
from langchain_core.messages import convert_to_messages
from langchain_openai import ChatOpenAI
from openai import OpenAI
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from src.app.web.app import create_app
from src.core.answer_schema import AnswerDocument
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.settings import AppSettings
from src.infra.slack_utils import create_slack_client


ANSWER_TEXT = "실제 FastAPI와 Python 런타임을 거친 한글 답변입니다. 🧪"


class LocalEmbeddings(Embeddings):
    """Replace the embedding service while retaining the real local index."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@contextmanager
def slack_server(requests: list[dict[str, object]]) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if "application/json" in self.headers.get("Content-Type", ""):
                payload = json.loads(raw)
            else:
                payload = {key: values[0] for key, values in parse_qs(raw.decode()).items()}
            requests.append({"path": path, "payload": payload})
            result = {
                "/users.lookupByEmail": {"ok": True, "user": {"id": "UINTENDED"}},
                "/conversations.open": {"ok": True, "channel": {"id": "DTARGET"}},
                "/chat.postMessage": {"ok": True, "channel": payload.get("channel", "DTARGET"), "ts": "1.0"},
            }.get(path, {"ok": False, "error": "unknown_method"})
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def run(port: int) -> None:
    release = Event()
    queries: list[str] = []
    submissions: list[dict[str, object]] = []
    slack_requests: list[dict[str, object]] = []
    held_queries: dict[str, Event] = {}

    def model_response(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        message_text = json.dumps(payload["messages"], ensure_ascii=False)
        if "제공자 오류 검증" in message_text:
            return httpx.Response(400, json={"error": {
                "message": "Fixture provider rejected the output schema.",
                "type": "invalid_request_error", "param": "response_format",
                "code": "invalid_json_schema",
            }})
        schema = payload.get("response_format", {}).get("json_schema", {}).get("name")
        if schema == "PlannerOutput":
            messages = convert_to_messages(payload["messages"])
            raw = next(message.content for message in messages if message.name == "request_context")
            context = json.loads(str(raw).split("\n", 1)[1])
            current_turn = context["user_turn_ledger"][-1]
            query = current_turn["text"]
            if query in held_queries and not held_queries[query].wait(20):
                raise AssertionError("The browser did not release its held question")
            if "대기 검증" in message_text and not release.wait(20):
                raise AssertionError("The browser did not release the in-flight request")
            contract = WireRequestContract(slack_recipient={"state": "omitted"})
            if "Slack으로 보내줘" in query:
                contract = WireRequestContract.model_validate({
                    "slack_recipient": {"state": "omitted"},
                    "actions": {"slack_notify": {"intent": "requested", "evidence_ids": ["send"]}},
                    "evidence": [{"id": "send", "turn_id": current_turn["turn_id"],
                                  "quote": "Slack으로 보내줘", "scope": "actions.slack_notify",
                                  "interpretation": "instruction"}],
                })
            files = context["upload_files"]
            content = InitialPlannerOutput.model_validate({
                "use_retrieval": bool(files),
                "tasks": [{"route": "upload", "query": "alpha beta value", "k": 4,
                           "requirement": {"file_ids": [item["file_id"] for item in files]}}] if files else [],
                "request_contract": contract.model_dump(mode="json"),
            }).model_dump_json()
        elif schema == "AnswerDocument":
            messages = convert_to_messages(payload["messages"])
            packet = json.loads(str(messages[-1].content).split("\n", 2)[2])
            blocks = [{"type": "code", "language": "python", "content": {
                "text": item["excerpt"], "basis": "excerpt", "refs": [item["id"]],
            }} for item in packet]
            content = AnswerDocument.model_validate({"blocks": blocks or [{
                "type": "paragraph",
                "content": [{"text": ANSWER_TEXT, "basis": "interaction", "refs": []}],
            }]}).model_dump_json()
        else:
            raise AssertionError(f"Unexpected provider schema in chat smoke: {schema}")
        return httpx.Response(200, json={
            "id": "chatcmpl-browser-fixture", "object": "chat.completion", "created": 1,
            "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": content,
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def decisions_response(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("POST", "/v1/decisions")
        payload = json.loads(request.content)
        return httpx.Response(200, json={
            "model": payload["model"],
            "answers": [{"type": "predicate", "name": item["name"], "probability": 0.99}
                        for item in payload["questions"]],
            "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32},
        })

    with (
        TemporaryDirectory(prefix="documate-react-e2e-") as temporary_root,
        pytest.MonkeyPatch.context() as patch,
        httpx.Client(transport=httpx.MockTransport(model_response)) as model_http,
        httpx.Client(transport=httpx.MockTransport(decisions_response)) as decisions_http,
        slack_server(slack_requests) as slack_url,
    ):
        settings = AppSettings(
            _env_file=None,
            openai_api_key="fixture-key", tavily_api_key="fixture-key",
            slack_bot_token="fixture-token", slack_default_user_id="UDEFAULT", slack_default_dm_email="",
            chat_model="gpt-4o-mini", planner_model="gpt-4o-mini", summary_model="gpt-4o-mini",
            synthesis_use_responses_api=False, docling_enabled=False,
            upload_search_mode="baseline", upload_max_file_mib=1, upload_max_total_mib=2,
        )
        patch.setattr("src.app.web.app.get_settings", lambda: settings)
        patch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: Path(temporary_root))
        patch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: LocalEmbeddings())
        patch.setenv("LANGSMITH_TRACING", "false")
        patch.setenv("LANGCHAIN_TRACING_V2", "false")
        patch.setenv("NO_PROXY", "127.0.0.1,localhost")

        def chat_model(**kwargs: object) -> ChatOpenAI:
            return ChatOpenAI(**kwargs, http_client=model_http, base_url="https://provider.test/v1")

        def decisions_client(**kwargs: object) -> OpenAI:
            return OpenAI(**kwargs, http_client=decisions_http, base_url="https://provider.test/v1")

        patch.setattr("src.infra.llm.ChatOpenAI", chat_model)
        patch.setattr("src.infra.llm.OpenAI", decisions_client)

        def local_slack_client(token: str | None):
            client = create_slack_client(token)
            if client is not None:
                client.base_url = slack_url
                client.timeout = 2
            return client

        patch.setattr("src.infra.tools.slack_notify.create_slack_client", local_slack_client)
        connect = socket.socket.connect

        def loopback_only(sock: socket.socket, address: object) -> None:
            if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1", "localhost"}:
                raise AssertionError("External network is forbidden in the React API fixture")
            connect(sock, address)

        patch.setattr(socket.socket, "connect", loopback_only)
        app = create_app()

        @app.middleware("http")
        async def record_submission(request: Request, call_next: RequestResponseEndpoint) -> Response:
            if request.method == "POST" and request.url.path == "/agent/stream":
                payload = await request.json()
                queries.append(payload["query"])
                submissions.append(payload)
            return await call_next(request)

        @app.get("/_fixture/ready")
        def ready() -> dict[str, bool]:
            return {"ready": True}

        @app.get("/_fixture/requests")
        def received() -> dict[str, object]:
            return {"queries": list(queries), "submissions": list(submissions), "slack": list(slack_requests)}

        @app.post("/_fixture/hold")
        async def hold_response(request: Request) -> dict[str, bool]:
            payload = await request.json()
            held_queries[payload["query"]] = Event()
            return {"held": True}

        @app.post("/_fixture/release")
        async def release_response(request: Request) -> dict[str, bool]:
            body = await request.body()
            if body:
                payload = json.loads(body)
                held_queries[payload["query"]].set()
            release.set()
            return {"released": True}

        @app.post("/_fixture/sessions/{session_id}/expire")
        def expire_session(session_id: str) -> dict[str, bool]:
            with app.state.session_store.locked_session(session_id) as (entry, _wait_ms):
                entry.agent.session.close()
            return {"expired": True}

        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", ws="none")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18000)
    run(parser.parse_args().port)
