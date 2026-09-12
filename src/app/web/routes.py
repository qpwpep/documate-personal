from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from src.infra.logging_utils import log_event
from src.infra.runtime_paths import get_save_text_output_dir
from src.app.web.cleanup import resolve_download_path
from src.app.web.schemas import AGENT_STREAM_EVENT_SCHEMAS, AgentRequest, AgentStreamEvent
from src.core.uploads import UploadManifest, UploadSyncRequest, UploadSyncResponse, validate_session_id


logger = logging.getLogger(__name__)
router = APIRouter()


def _upload_session_id(value: str) -> str:
    try:
        return validate_session_id(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/uploads", response_model=UploadManifest)
def get_upload_manifest(session_id: str, request: Request):
    return request.app.state.upload_service.get_manifest(_upload_session_id(session_id))


@router.post("/sessions/{session_id}/uploads/sync", response_model=UploadSyncResponse)
def sync_upload_manifest(session_id: str, request_data: UploadSyncRequest, request: Request):
    return request.app.state.upload_service.sync(_upload_session_id(session_id), request_data)


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
                "A valid final_response carries response, trace, debug and upload_manifest. "
                "The manifest is captured under the session lock after request execution. HTTP 200 and done "
                "alone do not indicate success. An error event may precede a final_response; "
                "preserve both the error and any final diagnostics. Clients must not automatically "
                "resubmit interrupted requests because agent actions may already have executed. "
                "x-sse-events maps each event name to its data schema."
            ),
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "x-sse-events": AGENT_STREAM_EVENT_SCHEMAS,
                    "example": (
                        'event: request_started\ndata: {"request_id":"abc12345","session_id":"demo"}\n\n'
                        'event: error\ndata: {"message":"UPLOAD_PATH_INVALID: Upload file not found"}\n\n'
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
async def download_file(filename: str):
    file_path = resolve_download_path(get_save_text_output_dir(), filename)

    if not file_path.exists():
        log_event(logger, logging.ERROR, "download_file_missing", path=file_path)
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="text/plain",
    )
