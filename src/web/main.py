import logging
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from langchain_core.messages import SystemMessage

from ..latency import LatencyBreakdownModel
from ..runtime_encoding import ensure_utf8_stdio
from .schemas import (
    AgentDebugInfo,
    PlannerDiagnostic,
    AgentRequest,
    AgentRetryContext,
    AgentResponse,
    AgentResponsePayload,
    AgentTokenUsage,
    EvidenceItem,
    RetrievalDiagnostic,
)
from ..agent_manager import AgentFlowManager
from ..settings import ConfigurationError, get_settings, validate_required_keys
from ..util.util import get_save_text_output_dir


ensure_utf8_stdio()

logger = logging.getLogger("uvicorn")


def _warn_if_utf8_mode_disabled() -> None:
    if sys.flags.utf8_mode == 1:
        return

    logger.warning(
        "UTF-8 mode is disabled (utf8_mode=0). For direct launch, run "
        "'uv run python -X utf8 -m uvicorn src.web.main:app --host 0.0.0.0 --port 8000' "
        "or set PYTHONUTF8=1 before startup. An already-started interpreter cannot fully "
        "switch utf8_mode at runtime."
    )


_warn_if_utf8_mode_disabled()
app = FastAPI()
ALLOWED_UPLOAD_SUFFIXES = {".py", ".ipynb"}


@dataclass
class SessionEntry:
    agent: AgentFlowManager
    last_accessed_monotonic: float
    created_monotonic: float


# In-memory per-session agent cache (TTL + LRU).
active_agents: dict[str, SessionEntry] = {}
_session_cache_lock = Lock()
_last_cleanup_monotonic = 0.0
_file_cleanup_lock = Lock()
_last_file_cleanup_monotonic = 0.0


@app.on_event("startup")
async def validate_app_settings_on_startup() -> None:
    try:
        settings = get_settings()
        validate_required_keys(settings, context="fastapi_startup")
    except ConfigurationError as exc:
        logger.error(str(exc))
        raise RuntimeError(str(exc)) from exc

    _run_file_cleanup_once(force=True, current_session_id=None)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    logger.info(f"[REQ ID: {request_id[:8]}] - Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"[REQ ID: {request_id[:8]}] - Finished request with status {response.status_code}")
    return response


def _get_or_create_agent(session_id: str) -> AgentFlowManager:
    """Return an AgentFlowManager for the session (TTL/LRU managed)."""
    settings = get_settings()
    now = time.monotonic()

    with _session_cache_lock:
        expired_removed_count = _maybe_run_cleanup(
            now=now,
            cleanup_interval_seconds=settings.session_cleanup_interval_seconds,
            ttl_seconds=settings.session_ttl_seconds,
        )

        existing_entry = active_agents.get(session_id)
        if existing_entry is not None:
            existing_entry.last_accessed_monotonic = now
            _log_session_cache_event(
                session_id=session_id,
                session_hit=True,
                session_recreated=False,
                expired_removed_count=expired_removed_count,
                lru_evicted_count=0,
                active_session_count=len(active_agents),
            )
            return existing_entry.agent

        recreated_agent = AgentFlowManager(settings=settings)
        active_agents[session_id] = SessionEntry(
            agent=recreated_agent,
            created_monotonic=now,
            last_accessed_monotonic=now,
        )
        lru_evicted_count = _evict_lru_if_needed(settings.max_active_sessions)
        active_session_count = len(active_agents)

        _log_session_cache_event(
            session_id=session_id,
            session_hit=False,
            session_recreated=True,
            expired_removed_count=expired_removed_count,
            lru_evicted_count=lru_evicted_count,
            active_session_count=active_session_count,
        )
        return active_agents[session_id].agent


def _log_session_cache_event(
    session_id: str,
    session_hit: bool,
    session_recreated: bool,
    expired_removed_count: int,
    lru_evicted_count: int,
    active_session_count: int,
) -> None:
    logger.info(
        "session_cache_event session_id=%s session_hit=%s session_recreated=%s "
        "expired_removed_count=%d lru_evicted_count=%d active_session_count=%d",
        session_id[:8],
        session_hit,
        session_recreated,
        expired_removed_count,
        lru_evicted_count,
        active_session_count,
    )


def _pop_session_entry(session_id: str) -> SessionEntry | None:
    entry = active_agents.pop(session_id, None)
    if entry is None:
        return None

    try:
        entry.agent.close()
    except Exception as exc:
        logger.warning("session_close_error session_id=%s error=%s", session_id[:8], exc)
    return entry


def _cleanup_expired_sessions(now: float, ttl_seconds: int) -> int:
    expired_session_ids = [
        sid
        for sid, entry in active_agents.items()
        if now - entry.last_accessed_monotonic > ttl_seconds
    ]
    for sid in expired_session_ids:
        _pop_session_entry(sid)
    return len(expired_session_ids)


def _evict_lru_if_needed(max_active_sessions: int) -> int:
    evicted_count = 0
    while len(active_agents) > max_active_sessions:
        lru_session_id = min(
            active_agents.items(),
            key=lambda item: item[1].last_accessed_monotonic,
        )[0]
        _pop_session_entry(lru_session_id)
        evicted_count += 1
    return evicted_count


def _maybe_run_cleanup(
    now: float,
    cleanup_interval_seconds: int,
    ttl_seconds: int,
) -> int:
    global _last_cleanup_monotonic
    if now - _last_cleanup_monotonic < cleanup_interval_seconds:
        return 0

    expired_removed_count = _cleanup_expired_sessions(now=now, ttl_seconds=ttl_seconds)
    _last_cleanup_monotonic = now
    return expired_removed_count


def _collect_protected_session_ids(current_session_id: str | None) -> set[str]:
    protected_session_ids: set[str] = set()

    with _session_cache_lock:
        protected_session_ids.update(active_agents.keys())

    if current_session_id:
        protected_session_ids.add(current_session_id)

    return protected_session_ids


def _get_latest_mtime_epoch(path: Path) -> float:
    latest_mtime = 0.0
    try:
        latest_mtime = path.stat().st_mtime
    except OSError:
        return latest_mtime

    if not path.is_dir():
        return latest_mtime

    try:
        children = path.rglob("*")
        for child in children:
            try:
                child_mtime = child.stat().st_mtime
                if child_mtime > latest_mtime:
                    latest_mtime = child_mtime
            except OSError:
                continue
    except OSError:
        return latest_mtime

    return latest_mtime


def _cleanup_expired_upload_dirs(
    now_epoch: float,
    ttl_seconds: int,
    protected_session_ids: set[str],
) -> dict[str, int]:
    uploads_root = Path("uploads")
    stats = {
        "scanned": 0,
        "deleted": 0,
        "skipped_active_dirs": 0,
        "errors": 0,
    }

    if not uploads_root.exists():
        return stats

    try:
        session_entries = list(uploads_root.iterdir())
    except OSError as exc:
        logger.warning("upload_cleanup_scan_error root=%s error=%s", uploads_root, exc)
        stats["errors"] += 1
        return stats

    for session_dir in session_entries:
        if not session_dir.is_dir():
            continue

        stats["scanned"] += 1
        session_id = session_dir.name

        if session_id in protected_session_ids:
            stats["skipped_active_dirs"] += 1
            continue

        latest_mtime = _get_latest_mtime_epoch(session_dir)
        if latest_mtime <= 0 or (now_epoch - latest_mtime) <= ttl_seconds:
            continue

        try:
            shutil.rmtree(session_dir)
            stats["deleted"] += 1
        except Exception as exc:
            logger.warning("upload_cleanup_error dir=%s error=%s", session_dir, exc)
            stats["errors"] += 1

    return stats


def _cleanup_expired_generated_files(now_epoch: float, ttl_seconds: int) -> dict[str, int]:
    output_dir = Path(get_save_text_output_dir())
    stats = {
        "scanned": 0,
        "deleted": 0,
        "errors": 0,
    }

    if not output_dir.exists():
        return stats

    try:
        txt_files = list(output_dir.glob("*.txt"))
    except OSError as exc:
        logger.warning("generated_file_scan_error root=%s error=%s", output_dir, exc)
        stats["errors"] += 1
        return stats

    for txt_file in txt_files:
        if not txt_file.is_file():
            continue

        stats["scanned"] += 1

        try:
            file_mtime = txt_file.stat().st_mtime
        except OSError as exc:
            logger.warning("generated_file_stat_error file=%s error=%s", txt_file, exc)
            stats["errors"] += 1
            continue

        if (now_epoch - file_mtime) <= ttl_seconds:
            continue

        try:
            txt_file.unlink()
            stats["deleted"] += 1
        except Exception as exc:
            logger.warning("generated_file_cleanup_error file=%s error=%s", txt_file, exc)
            stats["errors"] += 1

    return stats


def _run_file_cleanup_once(force: bool, current_session_id: str | None = None) -> dict[str, int | bool]:
    global _last_file_cleanup_monotonic

    settings = get_settings()
    now_monotonic = time.monotonic()

    result: dict[str, int | bool] = {
        "interval_skipped": False,
        "upload_dirs_scanned": 0,
        "upload_dirs_deleted": 0,
        "skipped_active_dirs": 0,
        "generated_files_scanned": 0,
        "generated_files_deleted": 0,
        "errors": 0,
    }

    try:
        with _file_cleanup_lock:
            interval_skipped = False
            if (
                not force
                and (now_monotonic - _last_file_cleanup_monotonic) < settings.file_cleanup_interval_seconds
            ):
                interval_skipped = True
                result["interval_skipped"] = True
            else:
                now_epoch = time.time()
                protected_session_ids = _collect_protected_session_ids(current_session_id)
                upload_stats = _cleanup_expired_upload_dirs(
                    now_epoch=now_epoch,
                    ttl_seconds=settings.session_ttl_seconds,
                    protected_session_ids=protected_session_ids,
                )
                generated_stats = _cleanup_expired_generated_files(
                    now_epoch=now_epoch,
                    ttl_seconds=settings.generated_file_ttl_seconds,
                )
                _last_file_cleanup_monotonic = now_monotonic

                errors = upload_stats["errors"] + generated_stats["errors"]
                result = {
                    "interval_skipped": interval_skipped,
                    "upload_dirs_scanned": upload_stats["scanned"],
                    "upload_dirs_deleted": upload_stats["deleted"],
                    "skipped_active_dirs": upload_stats["skipped_active_dirs"],
                    "generated_files_scanned": generated_stats["scanned"],
                    "generated_files_deleted": generated_stats["deleted"],
                    "errors": errors,
                }
    except Exception as exc:
        logger.warning("file_cleanup_unhandled_error force=%s error=%s", force, exc)
        result["errors"] = int(result.get("errors", 0)) + 1

    logger.info(
        "file_cleanup_event force=%s interval_skipped=%s upload_dirs_scanned=%d "
        "upload_dirs_deleted=%d skipped_active_dirs=%d generated_files_scanned=%d "
        "generated_files_deleted=%d errors=%d",
        force,
        result["interval_skipped"],
        result["upload_dirs_scanned"],
        result["upload_dirs_deleted"],
        result["skipped_active_dirs"],
        result["generated_files_scanned"],
        result["generated_files_deleted"],
        result["errors"],
    )
    return result


def _validate_upload_file_path(upload_file_path: str | None, session_id: str) -> str | None:
    """
    Validate client-provided upload path and return a normalized absolute path.
    The file must exist under uploads/<session_id>/ and be one of allowed types.
    """
    if not upload_file_path:
        return None

    try:
        session_upload_dir = (Path("uploads") / session_id).resolve()
        candidate_path = Path(upload_file_path).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid upload_file_path") from exc

    if session_upload_dir not in candidate_path.parents:
        raise HTTPException(status_code=400, detail="Invalid upload file location")

    if candidate_path.suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported upload file type")

    if not candidate_path.is_file():
        raise HTTPException(status_code=400, detail="Upload file not found")

    return str(candidate_path)


def _normalize_debug_info(raw_debug: dict | None, latency_ms_server: int | None) -> AgentDebugInfo:
    debug = raw_debug or {}
    tool_calls = debug.get("tool_calls") or []
    errors = debug.get("errors") or []
    observed_evidence_raw = debug.get("observed_evidence") or []

    token_usage_raw = debug.get("token_usage") or {}
    token_usage = AgentTokenUsage(
        prompt_tokens=int(token_usage_raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(token_usage_raw.get("completion_tokens", 0) or 0),
        total_tokens=int(token_usage_raw.get("total_tokens", 0) or 0),
    )

    observed_evidence: list[EvidenceItem] = []
    if isinstance(observed_evidence_raw, list):
        for item in observed_evidence_raw:
            if not isinstance(item, dict):
                continue
            try:
                observed_evidence.append(EvidenceItem.model_validate(item))
            except Exception:
                continue

    retry_context = None
    raw_retry_context = debug.get("retry_context")
    if isinstance(raw_retry_context, dict):
        try:
            retry_context = AgentRetryContext.model_validate(raw_retry_context)
        except Exception:
            retry_context = None

    retrieval_diagnostics: list[RetrievalDiagnostic] = []
    raw_retrieval_diagnostics = debug.get("retrieval_diagnostics")
    if isinstance(raw_retrieval_diagnostics, list):
        for item in raw_retrieval_diagnostics:
            if not isinstance(item, dict):
                continue
            try:
                retrieval_diagnostics.append(RetrievalDiagnostic.model_validate(item))
            except Exception:
                continue

    planner_diagnostics = None
    raw_planner_diagnostics = debug.get("planner_diagnostics")
    if isinstance(raw_planner_diagnostics, dict):
        try:
            planner_diagnostics = PlannerDiagnostic.model_validate(raw_planner_diagnostics)
        except Exception:
            planner_diagnostics = None

    latency_breakdown = None
    raw_latency_breakdown = debug.get("latency_breakdown")
    if isinstance(raw_latency_breakdown, dict):
        latency_payload = dict(raw_latency_breakdown)
        latency_payload["server_total_ms"] = latency_ms_server
        try:
            latency_breakdown = LatencyBreakdownModel.model_validate(latency_payload)
        except Exception:
            latency_breakdown = None

    return AgentDebugInfo(
        tool_calls=[str(name) for name in tool_calls if name],
        tool_call_count=int(debug.get("tool_call_count", len(tool_calls)) or len(tool_calls)),
        latency_ms_server=latency_ms_server,
        latency_breakdown=latency_breakdown,
        token_usage=token_usage,
        model_name=(str(debug.get("model_name")) if debug.get("model_name") else None),
        errors=[str(error) for error in errors if error],
        observed_evidence=observed_evidence,
        retry_context=retry_context,
        retrieval_diagnostics=retrieval_diagnostics,
        planner_diagnostics=planner_diagnostics,
    )


@app.post("/agent", response_model=AgentResponse)
async def run_agent_api(
    request: Request,
    request_data: AgentRequest,
):
    request_id = request.state.request_id[:8]

    user_query = request_data.query
    session_id = request_data.session_id
    _run_file_cleanup_once(force=False, current_session_id=session_id)
    upload_file_path = _validate_upload_file_path(request_data.upload_file_path, session_id)
    logger.info(f"[upload_file_path]: {upload_file_path}")

    agent_manager = _get_or_create_agent(session_id)

    # Inject Slack destination hints into the message stream (no auto-send here).
    slack_hints = []
    if request_data.slack_channel_id:
        slack_hints.append(f"channel_id={request_data.slack_channel_id}")
    if request_data.slack_user_id:
        slack_hints.append(f"user_id={request_data.slack_user_id}")
    if request_data.slack_email:
        slack_hints.append(f"email={request_data.slack_email}")

    if slack_hints:
        hint_text = (
            "[Slack Destinations]\n"
            + "\n".join(slack_hints)
            + "\n(When the user asks to send to Slack, call slack_notify with these values.)"
        )
        agent_manager.messages.append(SystemMessage(content=hint_text))

    logger.info(
        f"Session ID: {session_id[:8]} | [REQ ID: {request_id}] | "
        f"Agent Object ID: {id(agent_manager)} | Query: '{user_query[:20]}...'"
    )

    agent_started = time.monotonic()
    agent_answer = agent_manager.run_agent_flow(user_query, upload_file_path)
    latency_ms_server = int((time.monotonic() - agent_started) * 1000)

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

    debug_info = _normalize_debug_info(
        raw_debug=agent_answer.get("debug"),
        latency_ms_server=latency_ms_server,
    )

    logger.info(f"agent_answer: {agent_answer}")
    logger.info(f"answer: {answer}")
    logger.info(f"filepath: {file_path}")

    response = AgentResponse(
        response=response_payload,
        trace=f"Session ID: {session_id}, Request ID: {request_id}, Agent ID: {id(agent_manager)}",
        file_path=file_path,
        debug=debug_info if request_data.include_debug else None,
    )
    return response


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = _resolve_download_path(get_save_text_output_dir(), filename)

    if not file_path.exists():
        logger.error(f"Download request failed: File not found at {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="text/plain",
    )


def _resolve_download_path(output_dir: str, filename: str) -> Path:
    base_dir = Path(output_dir).resolve()
    normalized_filename = filename.replace("\\", "/")

    if Path(normalized_filename).is_absolute() or PureWindowsPath(normalized_filename).is_absolute():
        raise HTTPException(status_code=403, detail="Forbidden: Invalid file path")

    candidate = (base_dir / normalized_filename).resolve()
    try:
        candidate.relative_to(base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid file path") from exc
    return candidate
