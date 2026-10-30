from __future__ import annotations

import json
import logging
import re
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from src.infra.logging_utils import log_event
from src.infra.runtime_paths import get_save_text_output_dir
from src.infra.saved_artifacts import ArtifactError, read_saved_artifact
from src.app.web.cleanup import resolve_download_path
from src.app.web.schemas import (
    AGENT_STREAM_EVENT_SCHEMAS, AgentRequest, AgentStreamEvent,
    ClientConfig, ClientQueryConfig, ClientUploadConfig,
)
from src.core.contracts.outcome import AnswerExportResult
from src.core.conversation_memory import DEFAULT_QUERY_MAX_CHARS, DEFAULT_QUERY_MAX_UTF8_BYTES
from src.core.slack_contract import SlackDefault
from src.core.upload_formats import enabled_upload_suffixes
from src.core.uploads import UploadManifest, UploadSyncRequest, validate_session_id


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/client-config", response_model=ClientConfig)
def get_client_config(request: Request) -> ClientConfig:
    settings = request.app.state.settings
    return ClientConfig(
        upload=ClientUploadConfig(
            accepted_suffixes=sorted(enabled_upload_suffixes(docling_enabled=settings.docling_enabled)),
            max_files=settings.upload_max_files,
            max_file_bytes=settings.upload_max_file_mib * 1024 * 1024,
            max_total_bytes=settings.upload_max_total_mib * 1024 * 1024,
        ),
        query=ClientQueryConfig(
            max_chars=DEFAULT_QUERY_MAX_CHARS,
            max_utf8_bytes=DEFAULT_QUERY_MAX_UTF8_BYTES,
        ),
    )


@router.get("/slack/default-recipient", response_model=SlackDefault)
def get_slack_default_recipient(request: Request) -> SlackDefault:
    return request.app.state.settings.slack_default_recipient()


def _upload_session_id(value: str) -> str:
    try:
        return validate_session_id(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/uploads", response_model=UploadManifest)
def get_upload_manifest(session_id: str, request: Request):
    return request.app.state.upload_service.get_manifest(_upload_session_id(session_id))


@router.post("/sessions/{session_id}/uploads/sync", response_model=UploadManifest)
def sync_upload_manifest(session_id: str, request_data: UploadSyncRequest, request: Request):
    return request.app.state.upload_service.sync(_upload_session_id(session_id), request_data)


@router.post("/sessions/{session_id}/answers/{answer_id}/save", response_model=AnswerExportResult)
def save_answer(session_id: str, answer_id: str, request: Request):
    request_id = str(getattr(request.state, "request_id", None) or uuid4().hex)[:8]
    return request.app.state.answer_export_service.save(_upload_session_id(session_id), answer_id, request_id=request_id)


def _encode_sse_event(event: AgentStreamEvent) -> str:
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event}\ndata: {payload}\n\n"


@router.get("/")
async def root():
    return {"message": "Hello World"}


@router.post(
    "/agent/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "UTF-8 SSE frames: event: <name>\\ndata: <JSON object>\\n\\n. "
                "Each execution ends with one final_response carrying status, response, problem, message, "
                "request_id, trace, debug and upload_manifest. Status distinguishes completed, needs_input, "
                "failed, partial and refused; a failed result has no answer document. "
                "The manifest is captured under the session lock; null on failure means unknown state, not an empty attachment set. "
                "HTTP 200 and done alone do not indicate success. Clients must not automatically "
                "resubmit interrupted requests because agent actions may already have executed. "
                "x-sse-events maps each event name to its data schema."
            ),
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "x-sse-events": AGENT_STREAM_EVENT_SCHEMAS,
                    "example": (
                        'event: request_started\ndata: {"request_id":"abc12345","session_id":"demo"}\n\n'
                        'event: final_response\ndata: {"status":"failed","response":null,"message":"",'
                        '"problem":{"code":"internal_error","stage":"request","message":"요청을 완료하지 못했습니다.",'
                        '"next_action":"none"},"request_id":"abc12345","missing_slots":[],"trace":"Request ID: abc12345",'
                        '"debug":null,"upload_manifest":null}\n\n'
                        'event: done\ndata: {}\n\n'
                    ),
                },
            },
        },
    },
)
async def run_agent_stream_api(
    request: Request,
    request_data: AgentRequest,
):
    request_id = str(request.state.request_id)[:8]
    stream = request.app.state.agent_request_service.stream(
        request_id=request_id,
        request_data=request_data,
    )

    async def event_stream():
        async for event in stream:
            yield _encode_sse_event(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/download/{filename}")
def download_file(filename: str):
    output_dir = get_save_text_output_dir()
    file_path = resolve_download_path(output_dir, filename)
    if file_path.parent != output_dir.resolve():
        raise HTTPException(status_code=403, detail="Forbidden: Invalid file path")
    if file_path.suffix != ".txt":
        raise HTTPException(status_code=404, detail={"code": "manifest_missing", "message": "Saved artifact not found"})
    try:
        manifest, payload = read_saved_artifact(output_dir, filename)
    except ArtifactError as exc:
        status = {
            "manifest_missing": 404,
            "artifact_missing": 404,
            "artifact_expired": 410,
            "artifact_mismatch": 409,
        }.get(exc.code, 503)
        log_event(logger, logging.WARNING, "download_artifact_unavailable", filename=filename, code=exc.code)
        raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)}) from exc

    # Serve the bytes that were verified, rather than reopening a mutable path.
    return Response(
        content=payload,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{quote(_download_filename(payload), safe='')}",
            "ETag": f'"{manifest.artifact.sha256}"',
            "X-Artifact-Id": manifest.artifact.artifact_id,
            "X-Save-Binding-SHA256": manifest.operation.binding_sha256,
            "Cache-Control": "no-store",
        },
    )


def _download_filename(payload: bytes) -> str:
    """Give verified TXT bytes a readable name without changing artifact identity."""
    first_line = next((line.strip() for line in payload.decode("utf-8-sig").splitlines() if line.strip()), "")
    if first_line.startswith(("```", "~~~")):
        return "DocuMate 답변.txt"
    title = re.sub(r"\s*\[\d+\]", "", first_line.lstrip("# "))
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f`]', " ", title)
    title = " ".join(title.split())[:70].strip(" .-")
    return f"DocuMate - {title}.txt" if title else "DocuMate 답변.txt"
