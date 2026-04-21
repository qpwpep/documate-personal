from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class AgentRequestContext:
    fastapi_url: str
    session_id: str
    slack_user_id: str = ""
    slack_email: str = ""
    slack_channel_id: str = ""
    upload_file_path: str | None = None


@dataclass
class AgentCallResult:
    answer: str
    file_path: str | None = None
    evidence_items: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class AgentStreamEvent:
    event: str
    data: dict[str, Any] = field(default_factory=dict)
    result: AgentCallResult | None = None


def get_agent_response(user_input: str, context: AgentRequestContext) -> AgentCallResult:
    endpoint = f"{context.fastapi_url}/agent"
    try:
        resp = requests.post(endpoint, json=_build_payload(user_input, context), timeout=60)

        if resp.status_code == 200:
            return _parse_agent_response_data(resp.json())

        return AgentCallResult(
            answer=(
                f"Agent 호출 실패: 상태 코드 {resp.status_code}\n"
                f"응답: {resp.text}"
            ),
        )

    except requests.exceptions.Timeout:
        return AgentCallResult(
            answer="요청이 타임아웃되었습니다. 서버 상태를 확인해 주세요.",
        )
    except requests.exceptions.ConnectionError:
        return AgentCallResult(
            answer="FastAPI 서버에 연결할 수 없습니다. 서버(8000번 포트) 실행 여부를 확인해 주세요.",
        )
    except Exception as exc:
        return AgentCallResult(
            answer=f"요청 중 예기치 않은 오류가 발생했습니다: {exc}",
        )


def stream_agent_response(
    user_input: str,
    context: AgentRequestContext,
) -> Iterator[AgentStreamEvent]:
    endpoint = f"{context.fastapi_url}/agent/stream"
    payload = _build_payload(user_input, context)
    saw_event = False

    try:
        with requests.post(endpoint, json=payload, timeout=60, stream=True) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"stream endpoint returned {resp.status_code}")

            for event in _iter_sse_events(
                resp.iter_content(chunk_size=None, decode_unicode=True)
            ):
                saw_event = True
                yield event

            if not saw_event:
                raise RuntimeError("empty stream response")

    except Exception as exc:
        if saw_event:
            yield AgentStreamEvent(
                event="error",
                data={"message": f"스트리밍 중 오류가 발생했습니다: {exc}"},
            )
            return

        fallback_result = get_agent_response(user_input, context)
        yield _build_final_response_event(fallback_result)


def _build_payload(user_input: str, context: AgentRequestContext) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": user_input,
        "session_id": context.session_id,
    }

    if context.slack_user_id:
        payload["slack_user_id"] = context.slack_user_id
    if context.slack_email:
        payload["slack_email"] = context.slack_email
    if context.slack_channel_id:
        payload["slack_channel_id"] = context.slack_channel_id
    if context.upload_file_path:
        payload["upload_file_path"] = context.upload_file_path
    return payload


def _parse_agent_response_data(data: Any) -> AgentCallResult:
    payload = data if isinstance(data, dict) else {}
    response_payload = payload.get("response") or {}
    if isinstance(response_payload, dict):
        answer = str(response_payload.get("answer", "") or "")
        evidence = response_payload.get("evidence")
        evidence_items = evidence if isinstance(evidence, list) else []
    else:
        answer = str(response_payload)
        evidence_items = []

    return AgentCallResult(
        answer=answer,
        file_path=payload.get("file_path"),
        evidence_items=evidence_items,
    )


def _build_final_response_event(result: AgentCallResult) -> AgentStreamEvent:
    return AgentStreamEvent(
        event="final_response",
        data={
            "response": {
                "answer": result.answer,
                "claims": [],
                "evidence": result.evidence_items,
                "confidence": None,
            },
            "trace": "",
            "file_path": result.file_path,
            "debug": None,
        },
        result=result,
    )


def _iter_sse_events(chunks: Iterable[str | bytes]) -> Iterator[AgentStreamEvent]:
    buffer = ""
    event_name = "message"
    data_lines: list[str] = []

    for chunk in chunks:
        if chunk is None:
            continue
        if isinstance(chunk, bytes):
            text_chunk = chunk.decode("utf-8", errors="replace")
        else:
            text_chunk = str(chunk)
        if not text_chunk:
            continue
        buffer += text_chunk

        while True:
            newline_index = buffer.find("\n")
            if newline_index < 0:
                break
            line = buffer[:newline_index]
            buffer = buffer[newline_index + 1 :]
            if line.endswith("\r"):
                line = line[:-1]

            if not line:
                event = _finalize_sse_event(event_name, data_lines)
                if event is not None:
                    yield event
                event_name = "message"
                data_lines = []
                continue

            if line.startswith(":"):
                continue

            field, separator, value = line.partition(":")
            if not separator:
                continue
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                event_name = value or "message"
            elif field == "data":
                data_lines.append(value)

    if data_lines:
        event = _finalize_sse_event(event_name, data_lines)
        if event is not None:
            yield event


def _finalize_sse_event(event_name: str, data_lines: list[str]) -> AgentStreamEvent | None:
    if not data_lines and not event_name:
        return None

    payload: dict[str, Any] = {}
    if data_lines:
        raw_payload = "\n".join(data_lines)
        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError:
            parsed = {"message": raw_payload}
        if isinstance(parsed, dict):
            payload = parsed
        else:
            payload = {"value": parsed}

    result = _parse_agent_response_data(payload) if event_name == "final_response" else None
    return AgentStreamEvent(
        event=event_name or "message",
        data=payload,
        result=result,
    )
