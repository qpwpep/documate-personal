from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests
from urllib3.exceptions import ReadTimeoutError

from src.core.answer_schema import AnswerResponse
from src.core.uploads import UploadContext, UploadManifest, UploadSyncResponse
from src.infra.sse import iter_sse_events


@dataclass
class AgentRequestContext:
    fastapi_url: str
    session_id: str
    slack_user_id: str = ""
    slack_email: str = ""
    slack_channel_id: str = ""
    upload_file_path: str | None = None
    uploads: UploadContext | None = None


class UploadAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, code: str = "UPLOAD_REQUEST_FAILED", files: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.files = files or []


def _upload_request(method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        timeout = (5, 15) if method == "get" else (10, 180)
        with requests.request(method, endpoint, json=payload, timeout=timeout, allow_redirects=False) as response:
            try:
                data = response.json()
            except ValueError as exc:
                raise UploadAPIError("첨부 서버의 응답 형식을 확인할 수 없습니다.", status_code=response.status_code) from exc
            if response.status_code != 200:
                detail = data.get("detail", data) if isinstance(data, dict) else data
                if isinstance(detail, dict):
                    raise UploadAPIError(str(detail.get("message", "첨부 변경에 실패했습니다.")), status_code=response.status_code,
                                         code=str(detail.get("code", "UPLOAD_REQUEST_FAILED")), files=detail.get("files", []))
                raise UploadAPIError(str(detail), status_code=response.status_code)
            if not isinstance(data, dict):
                raise UploadAPIError("첨부 서버가 객체 형식의 응답을 반환하지 않았습니다.")
            return data
    except requests.RequestException as exc:
        raise UploadAPIError("첨부 서버와 연결하지 못했습니다. 처리 여부를 확인하거나 같은 변경을 다시 시도해 주세요.") from exc


def fetch_upload_manifest(fastapi_url: str, session_id: str) -> UploadManifest:
    endpoint = f"{fastapi_url.rstrip('/')}/sessions/{quote(session_id, safe='')}/uploads"
    try:
        return UploadManifest.model_validate(_upload_request("get", endpoint))
    except ValueError as exc:
        raise UploadAPIError("첨부 목록의 응답 형식이 올바르지 않습니다.") from exc


def sync_uploads(fastapi_url: str, session_id: str, payload: dict[str, Any]) -> UploadSyncResponse:
    endpoint = f"{fastapi_url.rstrip('/')}/sessions/{quote(session_id, safe='')}/uploads/sync"
    data = _upload_request("post", endpoint, payload)
    try:
        return UploadSyncResponse.model_validate(data)
    except (KeyError, ValueError, TypeError) as exc:
        raise UploadAPIError("첨부 변경 결과의 응답 형식이 올바르지 않습니다.") from exc


@dataclass
class AgentCallResult:
    response: AnswerResponse
    upload_manifest: UploadManifest | None = None


@dataclass(frozen=True)
class AgentStreamEvent:
    event: str
    data: dict[str, Any] = field(default_factory=dict)
    result: AgentCallResult | None = None


def stream_agent_response(
    user_input: str,
    context: AgentRequestContext,
) -> Iterator[AgentStreamEvent]:
    endpoint = f"{context.fastapi_url.rstrip('/')}/agent/stream"
    payload = _build_payload(user_input, context)
    saw_event = False
    saw_error = False

    # A disconnected client cannot know whether execution or an action completed.
    # Send once; retrying or falling back could repeat LLM calls and side effects.
    try:
        with requests.post(
            endpoint,
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=60,
            stream=True,
            allow_redirects=False,
        ) as resp:
            if resp.status_code != 200:
                yield _stream_error(
                    "http_error",
                    f"Agent 호출 실패: 상태 코드 {resp.status_code}\n응답: {resp.text}",
                    status_code=resp.status_code,
                )
                return

            content_type = resp.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "text/event-stream":
                raise ValueError(
                    f"Content-Type이 text/event-stream이어야 합니다. 수신: {content_type or '없음'}"
                )

            for event in _iter_sse_events(resp.iter_content(chunk_size=None)):
                saw_event = True
                saw_error = saw_error or event.event == "error"
                yield event
                if event.event == "final_response":
                    return
                if event.event == "done":
                    break

            if saw_error:
                return
            if not saw_event:
                yield _stream_error(
                    "empty_stream",
                    "스트리밍 응답이 비어 있습니다. 서버의 처리 결과를 확인해 주세요.",
                )
            else:
                yield _stream_error(
                    "missing_final_response",
                    "최종 응답을 받기 전에 스트림이 종료되었습니다. 서버의 처리 결과를 확인해 주세요.",
                )

    except (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    ) as exc:
        # requests.iter_content wraps urllib3 read timeouts in ConnectionError.
        if isinstance(exc, requests.exceptions.Timeout) or any(
            isinstance(reason, ReadTimeoutError) for reason in exc.args
        ):
            code = "timeout"
            message = "스트리밍 응답 대기 시간이 초과되었습니다."
        elif saw_event:
            code = "connection_interrupted"
            message = "스트리밍 응답을 받는 도중 연결이 끊어졌습니다."
        else:
            code = "connection_error"
            message = "첫 이벤트를 받기 전에 스트림 연결에 실패했습니다."
        yield _stream_error(
            code,
            f"{message} 서버에서 요청이 처리되었을 수 있으니 결과를 확인해 주세요.\n상세: {exc}",
        )
    except ValueError as exc:
        yield _stream_error("invalid_stream", f"스트리밍 응답 형식에 오류가 있습니다: {exc}")
    except Exception as exc:
        yield _stream_error("stream_error", f"스트리밍 응답 처리 중 오류가 발생했습니다: {exc}")


def _stream_error(code: str, message: str, **details: Any) -> AgentStreamEvent:
    return AgentStreamEvent(event="error", data={"code": code, "message": message, **details})


def _build_payload(user_input: str, context: AgentRequestContext) -> dict[str, Any]:
    if context.uploads is not None and context.upload_file_path is not None:
        raise ValueError("uploads와 upload_file_path를 동시에 보낼 수 없습니다.")
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
    if context.uploads is not None:
        payload["uploads"] = context.uploads.model_dump(mode="json")
    return payload


def _parse_agent_response_data(data: dict[str, Any]) -> AgentCallResult:
    response_payload = data.get("response")
    if not isinstance(response_payload, dict):
        raise ValueError("API 응답의 response는 객체여야 합니다.")

    manifest_payload = data.get("upload_manifest")
    return AgentCallResult(
        response=AnswerResponse.model_validate(response_payload),
        upload_manifest=UploadManifest.model_validate(manifest_payload) if manifest_payload is not None else None,
    )


def _iter_sse_events(chunks: Iterable[str | bytes]) -> Iterator[AgentStreamEvent]:
    for event in iter_sse_events(chunks):
        result = _parse_agent_response_data(event.data) if event.event == "final_response" else None
        yield AgentStreamEvent(event=event.event, data=event.data, result=result)
