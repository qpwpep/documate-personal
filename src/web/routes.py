from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from ..logging_utils import log_event
from ..runtime_paths import get_save_text_output_dir
from .cleanup import resolve_download_path
from .schemas import AgentRequest, AgentResponse, AgentStreamEvent


logger = logging.getLogger(__name__)
router = APIRouter()


def _encode_sse_event(event: AgentStreamEvent) -> str:
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event}\ndata: {payload}\n\n"


@router.get("/")
async def root():
    return {"message": "Hello World"}


@router.post("/agent", response_model=AgentResponse)
async def run_agent_api(
    request: Request,
    request_data: AgentRequest,
):
    request_id = str(request.state.request_id)[:8]
    result = await request.app.state.agent_request_service.run(
        request_id=request_id,
        request_data=request_data,
    )
    return result.to_response()


@router.post("/agent/stream")
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
