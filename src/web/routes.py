from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ..contracts import SessionMetadata
from ..contracts.debug import DEBUG_CRITICAL_FIELDS, DEBUG_REQUIRED_FIELDS, DEBUG_SCHEMA_VERSION
from ..contracts.boundary.debug import parse_llm_calls, parse_retry_state, parse_token_usage
from ..contracts.boundary.planner import parse_planner_diagnostic
from ..contracts.boundary.retrieval import parse_retrieval_diagnostics
from ..contracts.boundary.runtime import parse_slack_destination
from ..latency import LatencyBreakdownModel
from ..logging_utils import log_event
from ..runtime_paths import get_save_text_output_dir
from .cleanup import resolve_download_path, validate_upload_file_path
from .schemas import (
    AgentDebugInfo,
    AgentRequest,
    AgentResponse,
    AgentResponsePayload,
    AgentRetryContext,
    AgentTokenUsage,
    EvidenceItem,
    LLMCallMetadata,
    PlannerDiagnostic,
    RetrievalDiagnostic,
)


logger = logging.getLogger(__name__)
router = APIRouter()


def normalize_debug_info(raw_debug: dict | None, latency_ms_server: int | None) -> AgentDebugInfo:
    debug = raw_debug or {}
    present_keys = {str(key) for key in debug.keys()} if isinstance(debug, dict) else set()
    missing_required_debug_fields = [
        field for field in DEBUG_REQUIRED_FIELDS if field not in present_keys
    ]
    self_reported_missing_fields = [
        str(field_name)
        for field_name in (debug.get("missing_required_debug_fields") or [])
        if str(field_name).strip()
    ]
    for field_name in self_reported_missing_fields:
        if field_name not in missing_required_debug_fields:
            missing_required_debug_fields.append(field_name)
    critical_missing = [field for field in missing_required_debug_fields if field in DEBUG_CRITICAL_FIELDS]
    tool_calls = debug.get("tool_calls") or []
    errors = debug.get("errors") or []
    planner_errors_raw = debug.get("planner_errors") or []
    observed_evidence_raw = debug.get("observed_evidence") or []
    models_used_raw = debug.get("models_used")
    raw_llm_calls = debug.get("llm_calls")

    token_usage = parse_token_usage(debug.get("token_usage")) or AgentTokenUsage()

    observed_evidence: list[EvidenceItem] = []
    if isinstance(observed_evidence_raw, list):
        for item in observed_evidence_raw:
            if not isinstance(item, dict):
                continue
            try:
                observed_evidence.append(EvidenceItem.model_validate(item))
            except Exception:
                continue

    retry_context = parse_retry_state(debug.get("retry_context")) if debug.get("retry_context") else None
    retrieval_diagnostics = parse_retrieval_diagnostics(debug.get("retrieval_diagnostics"))
    planner_diagnostics = parse_planner_diagnostic(debug.get("planner_diagnostics"))
    llm_calls = parse_llm_calls(raw_llm_calls)

    models_used = [str(name) for name in models_used_raw if name] if isinstance(models_used_raw, list) else []
    if not models_used and llm_calls:
        for llm_call in llm_calls:
            model_name = llm_call.response_metadata.get("model_name") or llm_call.response_metadata.get("model")
            if model_name and str(model_name) not in models_used:
                models_used.append(str(model_name))

    latency_breakdown = None
    raw_latency_breakdown = debug.get("latency_breakdown")
    if isinstance(raw_latency_breakdown, dict):
        latency_payload = dict(raw_latency_breakdown)
        latency_payload["server_total_ms"] = latency_ms_server
        try:
            latency_breakdown = LatencyBreakdownModel.model_validate(latency_payload)
        except Exception:
            latency_breakdown = None

    schema_version_raw = debug.get("schema_version", DEBUG_SCHEMA_VERSION)
    try:
        schema_version = int(schema_version_raw)
    except (TypeError, ValueError):
        schema_version = DEBUG_SCHEMA_VERSION
    raw_observability_status = str(debug.get("observability_status") or "").strip().lower()
    if raw_observability_status not in {"ok", "degraded", "failed"}:
        raw_observability_status = "ok"

    return AgentDebugInfo(
        schema_version=schema_version,
        observability_status=(
            "failed"
            if critical_missing or raw_observability_status == "failed"
            else (
                "degraded"
                if raw_observability_status == "degraded" or missing_required_debug_fields
                else raw_observability_status
            )
        ),
        missing_required_debug_fields=missing_required_debug_fields,
        tool_calls=[str(name) for name in tool_calls if name],
        tool_call_count=int(debug.get("tool_call_count", len(tool_calls)) or len(tool_calls)),
        latency_ms_server=latency_ms_server,
        latency_breakdown=latency_breakdown,
        token_usage=token_usage,
        model_name=(str(debug.get("model_name")) if debug.get("model_name") else None),
        models_used=models_used,
        llm_calls=llm_calls,
        errors=[str(error) for error in errors if error],
        planner_errors=[str(error) for error in planner_errors_raw if error]
        if isinstance(planner_errors_raw, list)
        else [],
        observed_evidence=observed_evidence,
        retry_context=retry_context,
        retrieval_diagnostics=retrieval_diagnostics,
        planner_diagnostics=planner_diagnostics,
    )


def build_session_metadata_snapshot(request_data: AgentRequest) -> SessionMetadata:
    slack_destination = parse_slack_destination(
        {
            "channel_id": request_data.slack_channel_id,
            "user_id": request_data.slack_user_id,
            "email": request_data.slack_email,
        }
    )
    return SessionMetadata(
        slack_destination=slack_destination if slack_destination.has_destination() else None,
    )


@router.get("/")
async def root():
    return {"message": "Hello World"}


@router.post("/agent", response_model=AgentResponse)
async def run_agent_api(
    request: Request,
    request_data: AgentRequest,
):
    request_id = request.state.request_id[:8]
    cleaner = request.app.state.runtime_cleaner
    session_store = request.app.state.session_store

    user_query = request_data.query
    session_id = request_data.session_id
    cleaner.run_once(force=False, current_session_id=session_id)
    upload_file_path = validate_upload_file_path(request_data.upload_file_path, session_id)
    session_metadata = build_session_metadata_snapshot(request_data)
    agent_manager = session_store.get_or_create(session_id)

    log_event(
        logger,
        logging.INFO,
        "agent_request",
        session_id=session_id[:8],
        request_id=request_id,
        agent_id=id(agent_manager),
        query=user_query[:60],
        upload_file_path=upload_file_path,
    )

    started = time.monotonic()
    agent_manager, agent_answer, session_lock_wait_ms = await asyncio.to_thread(
        session_store.run_session_request,
        session_id=session_id,
        session_metadata=session_metadata,
        user_input=user_query,
        upload_file_path=upload_file_path,
    )
    latency_ms_server = int((time.monotonic() - started) * 1000)

    answer = str(agent_answer.get("message") or "")
    file_path = agent_answer.get("filepath", "")
    response_payload_raw = agent_answer.get("response_payload")

    fallback_payload = {
        "answer": answer,
        "claims": [],
        "evidence": [],
        "confidence": None,
    }
    payload_candidate = response_payload_raw if isinstance(response_payload_raw, dict) else fallback_payload
    try:
        response_payload = AgentResponsePayload.model_validate(payload_candidate)
    except Exception:
        response_payload = AgentResponsePayload.model_validate(fallback_payload)

    debug_info = normalize_debug_info(
        raw_debug=agent_answer.get("debug"),
        latency_ms_server=latency_ms_server,
    )

    log_event(
        logger,
        logging.INFO,
        "agent_response",
        session_id=session_id[:8],
        request_id=request_id,
        agent_id=id(agent_manager),
        latency_ms_server=latency_ms_server,
        session_lock_wait_ms=session_lock_wait_ms,
        session_lock_contended=session_lock_wait_ms > 0,
        file_path=file_path,
    )

    return AgentResponse(
        response=response_payload,
        trace=f"Session ID: {session_id}, Request ID: {request_id}, Agent ID: {id(agent_manager)}",
        file_path=file_path,
        debug=debug_info if request_data.include_debug else None,
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
