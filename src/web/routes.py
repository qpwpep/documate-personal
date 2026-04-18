from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ..logging_utils import log_event
from ..runtime_paths import get_save_text_output_dir
from .cleanup import resolve_download_path
from .schemas import AgentRequest, AgentResponse


logger = logging.getLogger(__name__)
router = APIRouter()


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
