from __future__ import annotations

import logging
import os
import sys
import time

from src.infra.logging_utils import log_event
from src.infra.runtime_encoding import build_utf8_env
from src.infra.runtime_paths import get_project_root_path, get_runtime_log_path
from src.infra.settings import ConfigurationError, get_settings, validate_required_keys
from src.app.web.static_ui import REACT_BUILD_INSTRUCTIONS, get_frontend_dist_path
from src.app.service_manager._bootstrap import logger
from src.app.service_manager import process_client
from src.app.service_manager import state as service_state


FASTAPI_LOG_FILE = "fastapi.log"
FASTAPI_PORT = 8000
FASTAPI_PROCESS_TOKENS = ["uvicorn", "src.app.web.app:app"]


def run_web_service(mode: str) -> int:
    if mode == "stopweb":
        return _stop_web_service()
    if mode == "statusweb":
        return _show_web_status()
    if mode not in {"startweb", "restartweb"}:
        log_event(logger, logging.ERROR, "unsupported_mode", mode=mode)
        return 1

    try:
        validate_required_keys(get_settings(), context=mode)
    except ConfigurationError as exc:
        log_event(logger, logging.ERROR, "configuration_error", context=mode, error=exc)
        return 1
    if not (get_frontend_dist_path() / "index.html").is_file():
        log_event(logger, logging.ERROR, "react_build_missing", path=get_frontend_dist_path(),
                  instructions=REACT_BUILD_INSTRUCTIONS)
        return 1

    # Validate the replacement before stopping the running server.
    if mode == "restartweb" and _stop_web_service() != 0:
        return 1
    return _start_web_service()


def _start_web_service() -> int:
    root = get_project_root_path()
    pid, _ = _resolve_managed_process(service_state.load_service_state())
    if pid is not None:
        log_event(logger, logging.WARNING, "service_already_running", pid=pid)
        return 1
    if process_client.is_port_open(FASTAPI_PORT):
        log_event(logger, logging.ERROR, "port_unavailable", port=FASTAPI_PORT)
        return 1

    utf8_env = build_utf8_env(os.environ.copy())
    fastapi_log_path = get_runtime_log_path(FASTAPI_LOG_FILE)
    fastapi_command = [sys.executable, "-X", "utf8", "-m", "uvicorn", "src.app.web.app:app",
                       "--host", "0.0.0.0", "--port", str(FASTAPI_PORT)]
    fastapi_process = None
    fastapi_created = None
    try:
        fastapi_process = process_client.start_background_process(
            command=fastapi_command, cwd=root, log_path=fastapi_log_path, env=utf8_env,
        )
        fastapi_created = process_client.get_process_create_time(fastapi_process.pid)
        if not process_client.wait_for_port_open(FASTAPI_PORT):
            raise RuntimeError(f"FastAPI port({FASTAPI_PORT}) was not opened. Log: {fastapi_log_path}")

        service_state.save_service_state(service_state.ServiceState(
            fastapi_pid=fastapi_process.pid, fastapi_create_time=fastapi_created,
            fastapi_log=str(fastapi_log_path), fastapi_port=FASTAPI_PORT,
            started_at_unix=int(time.time()), platform=sys.platform,
        ))
        log_event(logger, logging.INFO, "service_running",
                  fastapi_url=f"http://127.0.0.1:{FASTAPI_PORT}", ui_url=f"http://127.0.0.1:{FASTAPI_PORT}/app/")
        return 0
    except (OSError, RuntimeError) as exc:
        log_event(logger, logging.ERROR, "service_start_failed", error=exc)
        if fastapi_process is not None:
            process_client.terminate_process_tree(fastapi_process.pid, "FastAPI(launcher)", fastapi_created)
        service_state.remove_service_state()
        return 1


def _stop_web_service() -> int:
    state = service_state.load_service_state()
    if state is None:
        log_event(logger, logging.INFO, "service_stop_skipped", reason="no_managed_state")
        return 0
    pid, created = _resolve_managed_process(state)
    if pid is not None and not process_client.terminate_process_tree(pid, "FastAPI", created):
        return 1
    service_state.remove_service_state()
    return 0


def _show_web_status() -> int:
    pid, _ = _resolve_managed_process(service_state.load_service_state())
    running = pid is not None
    log_event(logger, logging.INFO, "web_service_status",
              status="running" if running else "stopped", fastapi_pid=pid,
              ui_url=f"http://127.0.0.1:{FASTAPI_PORT}/app/" if running else None)
    return 0 if running else 1


def _resolve_managed_process(state: service_state.ServiceState | None) -> tuple[int | None, float | None]:
    if state is None:
        return None, None
    return process_client.resolve_service_process(
        name="FastAPI", pid=state.fastapi_pid, create_time=state.fastapi_create_time,
        expected_tokens=FASTAPI_PROCESS_TOKENS, expected_cwd=get_project_root_path(),
    )
