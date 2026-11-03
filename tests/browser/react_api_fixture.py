"""Serve the real FastAPI and graph with isolated files and fake provider HTTP.

Only model, embedding and Slack service boundaries are replaced. Production
routes, sessions, upload storage, agent runner, exports and SSE encoding are real.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
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
from src.infra import llm_boundary
from src.infra.runtime_paths import get_upload_session_dir
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
    held_decisions: dict[str, tuple[Event, Event]] = {}

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
            if "전체 시간 소진 검증" in query:
                # Expire only this request while the fake provider is handling it.
                # The real post-call deadline check owns the resulting failure.
                deadline = llm_boundary._request_deadline.get()
                assert deadline is not None
                deadline.expires_at = time.monotonic() - 1
            if query in held_queries and not held_queries[query].wait(20):
                raise AssertionError("The browser did not release its held question")
            if "대기 검증" in message_text and not release.wait(20):
                raise AssertionError("The browser did not release the in-flight request")
            contract = WireRequestContract(slack_recipient={"state": "omitted"})
            copy_prefix = "다음 Markdown 원문을 그대로 복사해줘.\n\n"
            if query.startswith(copy_prefix):
                contract = WireRequestContract.model_validate({
                    "body": {"kind": "copy_input", "source": {
                        "turn_id": current_turn["turn_id"], "quote": query[len(copy_prefix):],
                    }},
                    "slack_recipient": {"state": "omitted"},
                })
            elif query == "방금 답변을 그대로 저장해줘. 한글 답변을 유지해줘.":
                contract = WireRequestContract.model_validate({
                    "body": {"kind": "copy_answer", "source": {"ref": "previous"}},
                    "actions": {"save_text": {"intent": "requested", "evidence_ids": ["export"]}},
                    "answer": {"content": [{"kind": "custom", "mode": "required",
                                            "description": "한글 답변 유지", "evidence_ids": ["export"]}]},
                    "evidence": [{"id": "export", "turn_id": current_turn["turn_id"],
                                  "quote": query, "scope": "current_request", "interpretation": "instruction"}],
                    "slack_recipient": {"state": "omitted"},
                })
            elif "Slack으로 보내줘" in query:
                contract = WireRequestContract.model_validate({
                    "slack_recipient": {"state": "omitted"},
                    "actions": {"slack_notify": {"intent": "requested", "evidence_ids": ["send"]}},
                    "evidence": [{"id": "send", "turn_id": current_turn["turn_id"],
                                  "quote": "Slack으로 보내줘", "scope": "actions.slack_notify",
                                  "interpretation": "instruction"}],
                })
            files = context["upload_files"]
            tasks = [{"route": "upload", "query": "alpha beta value", "k": 4,
                      "requirement": {"file_ids": [item["file_id"] for item in files]}}] if files else []
            if "근거 부족 구분 검증" in query:
                tasks = [{"route": "upload", "requirement_id": symbol, "query": symbol, "k": 4,
                          "requirement": {"symbols": [symbol], "match": "symbol",
                                          "file_ids": [item["file_id"] for item in files]}}
                         for symbol in ("alpha", "beta")]
            content = InitialPlannerOutput.model_validate({
                "use_retrieval": bool(tasks),
                "tasks": tasks,
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
        for query, (entered, released) in held_decisions.items():
            if query in payload["input"] and any(item["name"] == "request_ready" for item in payload["questions"]):
                entered.set()
                if not released.wait(20):
                    raise AssertionError("The browser did not release its held pre-validation request")
        probability = 0.91 if "한글 답변 유지" in payload["input"] else 0.99
        return httpx.Response(200, json={
            "model": payload["model"],
            "answers": [{"type": "predicate", "name": item["name"], "probability": (
                0.1 if "답변 조건 실패 검증" in payload["input"] and item["name"] == "request_fulfilled"
                else probability
            )}
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

        @app.get("/_fixture/sessions/{session_id}/staging")
        def staging_usage(session_id: str) -> dict[str, int]:
            staging = get_upload_session_dir(session_id) / "staging"
            sizes = [path.stat().st_size for path in staging.rglob("*") if path.is_file()]
            return {"files": len(sizes), "bytes": sum(sizes)}

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

        @app.post("/_fixture/decisions/hold")
        async def hold_decisions(request: Request) -> dict[str, bool]:
            payload = await request.json()
            held_decisions[payload["query"]] = Event(), Event()
            return {"held": True}

        @app.get("/_fixture/decisions")
        def decision_barriers() -> dict[str, bool]:
            return {query: entered.is_set() for query, (entered, _released) in held_decisions.items()}

        @app.post("/_fixture/decisions/release")
        async def release_decisions(request: Request) -> dict[str, bool]:
            payload = await request.json()
            held_decisions[payload["query"]][1].set()
            return {"released": True}

        @app.post("/_fixture/sessions/{session_id}/age-idle")
        def age_idle_session(session_id: str) -> dict[str, object]:
            store = app.state.session_store
            with store._lock:
                entry = store.active_agents[session_id]
                if entry.active_request_count or entry.request_lock.locked():
                    raise AssertionError("Only an idle session can be aged by this fixture")
                now = time.monotonic()
                entry.last_accessed_monotonic = now - settings.session_ttl_seconds - 1
                # Keep the periodic sweep in its cooldown. The next operation
                # must enforce this session's TTL without relying on that sweep.
                store._last_cleanup_monotonic = now
                return {"epoch": entry.agent.session.upload_manifest().epoch}

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
