import logging
import os
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


@app.on_event("startup")
async def validate_app_settings_on_startup() -> None:
    try:
        validate_required_keys(get_settings(), context="fastapi_startup")
    except ConfigurationError as exc:
        logger.error(str(exc))
        raise RuntimeError(str(exc)) from exc


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
