from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import quote

import requests
from pydantic import model_validator
from urllib3.exceptions import ReadTimeoutError

from src.app.uploads import UploadStageResult, stage_uploaded_files
from src.core.contracts.outcome import TurnResult
from src.core.slack_contract import RecipientSelector
from src.core.uploads import UploadContext, UploadManifest, UploadSyncRequest
from src.infra.sse import iter_sse_events


@dataclass
class AgentRequestContext:
    fastapi_url: str
    session_id: str
    slack_recipient: RecipientSelector | None = None
    include_debug: bool = False
    timeout_seconds: float = 60


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


def sync_uploads(fastapi_url: str, session_id: str, payload: dict[str, Any]) -> UploadManifest:
    endpoint = f"{fastapi_url.rstrip('/')}/sessions/{quote(session_id, safe='')}/uploads/sync"
    data = _upload_request("post", endpoint, payload)
    try:
        return UploadManifest.model_validate(data)
    except (KeyError, ValueError, TypeError) as exc:
        raise UploadAPIError("첨부 변경 결과의 응답 형식이 올바르지 않습니다.") from exc


class AgentCallResult(TurnResult):
    upload_manifest: UploadManifest | None = None

    @model_validator(mode="after")
    def require_confirmed_manifest(self):
        if self.status not in {"failed", "refused"} and self.upload_manifest is None:
            raise ValueError("A completed, partial or needs_input result requires upload_manifest")
        return self


@dataclass(frozen=True)
class AgentTransportObservation:
    """Transport facts captured at reception, before final-answer validation."""

    http_status: int | None = None
    request_id: str | None = None
    elapsed_ms: float = 0
    stream_opened: bool = False
    final_received: bool = False
    error_source: str | None = None
    error_type: str | None = None
    exception_type: str | None = None
    exception_detail: str | None = None
    http_body: str | None = None


@dataclass(frozen=True)
class AgentStreamEvent:
    event: str
    data: dict[str, Any] = field(default_factory=dict)
    result: AgentCallResult | None = None
    observation: AgentTransportObservation = field(default_factory=AgentTransportObservation)


def stream_agent_response(
    user_input: str,
    context: AgentRequestContext,
    *,
    uploads: UploadContext,
) -> Iterator[AgentStreamEvent]:
    endpoint = f"{context.fastapi_url.rstrip('/')}/agent/stream"
    payload = build_agent_payload(user_input, context, uploads=uploads)
    saw_event = False
    saw_error = False
    started = perf_counter()
    observation = AgentTransportObservation()

    def observed_error(code: str, message: str, *, exc: Exception | None = None,
                       error_type: str | None = None, **details: Any) -> AgentStreamEvent:
        return AgentStreamEvent(
            event="error", data={"code": code, "message": message, **details},
            observation=replace(
                observation, elapsed_ms=(perf_counter() - started) * 1000,
                error_source="client", error_type=error_type or code,
                exception_type=type(exc).__name__ if exc else None,
                exception_detail=str(exc) if exc else None,
            ),
        )

    # A disconnected client cannot know whether execution or an action completed.
    # Send once; retrying or falling back could repeat LLM calls and side effects.
    try:
        with requests.post(
            endpoint,
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=context.timeout_seconds,
            stream=True,
            allow_redirects=False,
        ) as resp:
            observation = replace(
                observation, http_status=resp.status_code,
                request_id=resp.headers.get("x-request-id") or resp.headers.get("X-Request-ID"),
            )
            if resp.status_code != 200:
                observation = replace(observation, http_body=resp.text)
                yield observed_error(
                    "http_error",
                    f"Agent 호출 실패: 상태 코드 {resp.status_code}. 오류 식별자를 확인해 주세요.",
                    status_code=resp.status_code,
                )
                return

            content_type = resp.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "text/event-stream":
                raise ValueError(
                    f"Content-Type이 text/event-stream이어야 합니다. 수신: {content_type or '없음'}"
                )

            observation = replace(observation, stream_opened=True)
            for raw_event in iter_sse_events(resp.iter_content(chunk_size=None)):
                saw_event = True
                observation = replace(
                    observation,
                    request_id=observation.request_id or raw_event.data.get("request_id"),
                    elapsed_ms=(perf_counter() - started) * 1000,
                    final_received=raw_event.event == "final_response",
                    error_source="server" if raw_event.event == "error" else None,
                    error_type="sse_error" if raw_event.event == "error" else None,
                )
                try:
                    event = _validated_event(raw_event.event, raw_event.data, observation)
                except ValueError as exc:
                    # Preserve the malformed payload for diagnostics while rejecting it
                    # as a usable answer in every client, including benchmarks.
                    yield AgentStreamEvent(
                        event="error",
                        data={"code": "invalid_stream", "message": "스트리밍 응답 형식을 검증하지 못했습니다. 서버의 처리 결과를 확인해 주세요.",
                              "raw_final_response": raw_event.data},
                        observation=replace(observation, error_source="client", error_type="agent_schema_error",
                                            exception_type=type(exc).__name__, exception_detail=str(exc)),
                    )
                    return
                saw_error = saw_error or event.event == "error"
                yield event
                if event.event == "final_response":
                    return
                if event.event == "done":
                    break

            if saw_error:
                return
            if not saw_event:
                yield observed_error(
                    "empty_stream",
                    "스트리밍 응답이 비어 있습니다. 서버의 처리 결과를 확인해 주세요.",
                )
            else:
                yield observed_error(
                    "missing_final_response",
                    "최종 응답을 받기 전에 스트림이 종료되었습니다. 서버의 처리 결과를 확인해 주세요.",
                )

    except (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        ReadTimeoutError,
    ) as exc:
        # requests.iter_content wraps urllib3 read timeouts in ConnectionError.
        if isinstance(exc, (requests.exceptions.Timeout, ReadTimeoutError)) or any(
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
        yield observed_error(
            code,
            f"{message} 서버에서 요청이 처리되었을 수 있으니 결과를 확인해 주세요.",
            exc=exc,
        )
    except ValueError as exc:
        yield observed_error("invalid_stream", "스트리밍 응답 형식에 오류가 있습니다. 서버의 처리 결과를 확인해 주세요.",
                             exc=exc, error_type="stream_parse_error")
    except Exception as exc:
        yield observed_error("stream_error", "스트리밍 응답 처리 중 오류가 발생했습니다. 서버의 처리 결과를 확인해 주세요.", exc=exc)


def build_agent_payload(
    user_input: str, context: AgentRequestContext, *, uploads: UploadContext,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": user_input,
        "session_id": context.session_id,
        "uploads": uploads.model_dump(mode="json"),
    }

    if context.slack_recipient is not None:
        payload["slack_recipient"] = RecipientSelector.model_validate(context.slack_recipient).model_dump(mode="json")
    if context.include_debug:
        payload["include_debug"] = True
    return payload


def _parse_agent_response_data(data: dict[str, Any]) -> AgentCallResult:
    return AgentCallResult.model_validate({
        key: value for key, value in data.items() if key in AgentCallResult.model_fields
    })


def _validated_event(event: str, data: dict[str, Any],
                     observation: AgentTransportObservation | None = None) -> AgentStreamEvent:
    result = _parse_agent_response_data(data) if event == "final_response" else None
    return AgentStreamEvent(event=event, data=data, result=result,
                            observation=observation or AgentTransportObservation())


class AgentSessionClient:
    """Shared user-session flow for confirmed attachments and streamed questions.

    Callers own presentation, scenario ordering and explicit upload retries. This
    client never retries a question or guesses state after an incomplete response.
    A None manifest means no usable server-confirmed snapshot, not empty attachments.
    """

    def __init__(self, context: AgentRequestContext, *, manifest: UploadManifest | None = None):
        self.context = context
        self.manifest = manifest

    def refresh_uploads(self) -> UploadManifest:
        # Failed refreshes must not leave stale confirmation usable by a question.
        self.manifest = None
        self.manifest = fetch_upload_manifest(self.context.fastapi_url, self.context.session_id)
        return self.manifest

    def stage_files(self, files: list[Any], session_path: Path, *, max_files: int,
                    max_file_mib: int, max_total_mib: int) -> UploadStageResult:
        if self.manifest is None:
            self.refresh_uploads()
        return stage_uploaded_files(
            files, session_path, existing_files=self.manifest.files,
            max_files=max_files, max_file_mib=max_file_mib, max_total_mib=max_total_mib,
        )

    def sync_uploads(self, request: UploadSyncRequest) -> UploadManifest:
        try:
            result = sync_uploads(self.context.fastapi_url, self.context.session_id, request.model_dump(mode="json"))
        except UploadAPIError:
            # A failed or lost response may hide a committed mutation or a newer
            # revision. Retain no confirmation that could be reused by a question.
            self.manifest = None
            raise
        self.manifest = result
        return result

    def upload_context(self) -> UploadContext:
        if self.manifest is None:
            self.refresh_uploads()
        return self.manifest.context()

    def stream(self, user_input: str) -> Iterator[AgentStreamEvent]:
        uploads = self.upload_context()
        received_valid_final = False
        try:
            for event in stream_agent_response(user_input, self.context, uploads=uploads):
                if event.event == "final_response" and event.result is not None:
                    received_valid_final = True
                    # Null means no confirmed snapshot, never "attachments were
                    # cleared". A later request must refresh before submission.
                    self.manifest = event.result.upload_manifest
                yield event
        finally:
            if not received_valid_final:
                self.manifest = None
