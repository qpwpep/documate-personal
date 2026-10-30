"""Serve the real FastAPI and graph with isolated files and fake provider HTTP.

Only the model-provider boundary is replaced. This fixture does not replace the
production routes, session store, agent runner, final response, or SSE encoding.
"""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import httpx
import pytest
import uvicorn
from fastapi import Request
from langchain_openai import ChatOpenAI
from openai import OpenAI
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from src.app.web.app import create_app
from src.core.answer_schema import AnswerDocument
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.settings import AppSettings


ANSWER_TEXT = "실제 FastAPI와 Python 런타임을 거친 한글 답변입니다. 🧪"


def run(port: int) -> None:
    release = Event()
    queries: list[str] = []

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
            if "대기 검증" in message_text and not release.wait(20):
                raise AssertionError("The browser did not release the in-flight request")
            content = InitialPlannerOutput.model_validate({
                "use_retrieval": False,
                "tasks": [],
                "request_contract": WireRequestContract(
                    slack_recipient={"state": "omitted"},
                ).model_dump(mode="json"),
            }).model_dump_json()
        elif schema == "AnswerDocument":
            content = AnswerDocument.model_validate({"blocks": [{
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
    ):
        settings = AppSettings(
            _env_file=None,
            openai_api_key="fixture-key", tavily_api_key="fixture-key",
            slack_bot_token="", slack_default_user_id="", slack_default_dm_email="",
            chat_model="gpt-4o-mini", planner_model="gpt-4o-mini", summary_model="gpt-4o-mini",
            synthesis_use_responses_api=False, docling_enabled=False,
        )
        patch.setattr("src.app.web.app.get_settings", lambda: settings)
        patch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: Path(temporary_root))
        patch.setenv("LANGSMITH_TRACING", "false")
        patch.setenv("LANGCHAIN_TRACING_V2", "false")
        patch.setenv("NO_PROXY", "127.0.0.1,localhost")

        def chat_model(**kwargs: object) -> ChatOpenAI:
            return ChatOpenAI(**kwargs, http_client=model_http, base_url="https://provider.test/v1")

        def decisions_client(**kwargs: object) -> OpenAI:
            return OpenAI(**kwargs, http_client=decisions_http, base_url="https://provider.test/v1")

        patch.setattr("src.infra.llm.ChatOpenAI", chat_model)
        patch.setattr("src.infra.llm.OpenAI", decisions_client)
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
            return await call_next(request)

        @app.get("/_fixture/ready")
        def ready() -> dict[str, bool]:
            return {"ready": True}

        @app.get("/_fixture/requests")
        def received() -> dict[str, list[str]]:
            return {"queries": list(queries)}

        @app.post("/_fixture/release")
        def release_response() -> dict[str, bool]:
            release.set()
            return {"released": True}

        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", ws="none")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18000)
    run(parser.parse_args().port)
