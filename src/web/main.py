import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from langchain_core.messages import SystemMessage

from .schemas import AgentRequest, AgentResponse
from ..agent_manager import AgentFlowManager
from ..settings import ConfigurationError, get_settings, validate_required_keys
from ..util.util import get_save_text_output_dir

logger = logging.getLogger("uvicorn")
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


def _cleanup_expired_sessions(now: float, ttl_seconds: int) -> int:
    expired_session_ids = [
        sid
        for sid, entry in active_agents.items()
        if now - entry.last_accessed_monotonic > ttl_seconds
    ]
    for sid in expired_session_ids:
        active_agents.pop(sid, None)
    return len(expired_session_ids)


def _evict_lru_if_needed(max_active_sessions: int) -> int:
    evicted_count = 0
    while len(active_agents) > max_active_sessions:
        lru_session_id = min(
            active_agents.items(),
            key=lambda item: item[1].last_accessed_monotonic,
        )[0]
        active_agents.pop(lru_session_id, None)
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

    agent_answer = agent_manager.run_agent_flow(user_query, upload_file_path)

    answer = agent_answer.get("message")
    file_path = agent_answer.get("filepath", "")

    logger.info(f"agent_answer: {agent_answer}")
    logger.info(f"answer: {answer}")
    logger.info(f"filepath: {file_path}")

    response = AgentResponse(
        response=answer,
        trace=f"Session ID: {session_id}, Request ID: {request_id}, Agent ID: {id(agent_manager)}",
        file_path=file_path,
    )
    return response


@app.get("/download/{filename}")
async def download_file(filename: str):
    output_dir = get_save_text_output_dir()
    file_path = os.path.join(output_dir, filename)

    if not os.path.realpath(file_path).startswith(os.path.realpath(output_dir)):
        raise HTTPException(status_code=403, detail="Forbidden: Invalid file path")

    if not os.path.exists(file_path):
        logger.error(f"Download request failed: File not found at {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/plain",
    )
