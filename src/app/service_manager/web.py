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
STREAMLIT_LOG_FILE = "streamlit.log"
FASTAPI_PORT = 8000
STREAMLIT_PORT = 8501

FASTAPI_PROCESS_TOKENS = ["uvicorn", "src.app.web.app:app"]
STREAMLIT_PROCESS_TOKENS = ["streamlit", "src/app/web/streamlit_app.py"]


def run_web_service(mode: str, *, ui: str | None = None) -> int:
    if mode == "stopweb":
        return _stop_web_services()
    if mode == "statusweb":
        return _show_web_status()
    if mode not in {"startweb", "restartweb"} or ui not in {None, "react", "streamlit"}:
        log_event(logger, logging.ERROR, "unsupported_mode", mode=mode, ui=ui)
        return 1

    previous = service_state.load_service_state() if mode == "restartweb" else None
    selected_ui = ui or (previous.ui if previous is not None else "react")
    try:
        validate_required_keys(get_settings(), context=mode)
    except ConfigurationError as exc:
        log_event(logger, logging.ERROR, "configuration_error", context=mode, error=exc)
        return 1
    if selected_ui == "react" and not (get_frontend_dist_path() / "index.html").is_file():
        log_event(logger, logging.ERROR, "react_build_missing", path=get_frontend_dist_path(),
                  instructions=REACT_BUILD_INSTRUCTIONS)
        return 1

    # Validate the requested replacement before stopping a working fallback UI.
    if mode == "restartweb" and _stop_web_services() != 0:
        return 1
    return _start_web_services(selected_ui)


def _start_web_services(ui: str) -> int:
    root = get_project_root_path()
    running = [f"{name}(pid={pid})" for name, (pid, _) in
               _resolve_managed_processes(service_state.load_service_state()).items() if pid is not None]
    if running:
        log_event(logger, logging.WARNING, "services_already_running", services=", ".join(running))
        return 1

    ports = [("FastAPI", FASTAPI_PORT)]
    if ui == "streamlit":
        ports.append(("Streamlit", STREAMLIT_PORT))
    occupied = [f"{name}:{port}" for name, port in ports if process_client.is_port_open(port)]
    if occupied:
        log_event(logger, logging.ERROR, "ports_unavailable", ports=", ".join(occupied))
        return 1

    utf8_env = build_utf8_env(os.environ.copy())
    fastapi_log_path = get_runtime_log_path(FASTAPI_LOG_FILE)
    streamlit_log_path = get_runtime_log_path(STREAMLIT_LOG_FILE)
    fastapi_command = [sys.executable, "-X", "utf8", "-m", "uvicorn", "src.app.web.app:app",
                       "--host", "0.0.0.0", "--port", str(FASTAPI_PORT)]
    streamlit_command = [sys.executable, "-X", "utf8", "-m", "streamlit", "run",
                        "src/app/web/streamlit_app.py", "--server.port", str(STREAMLIT_PORT)]
    fastapi_process = None
    streamlit_process = None
    fastapi_created = None
    streamlit_created = None
    try:
        fastapi_process = process_client.start_background_process(
            command=fastapi_command, cwd=root, log_path=fastapi_log_path, env=utf8_env,
        )
        fastapi_created = process_client.get_process_create_time(fastapi_process.pid)
        if not process_client.wait_for_port_open(FASTAPI_PORT):
            raise RuntimeError(f"FastAPI port({FASTAPI_PORT}) was not opened. Log: {fastapi_log_path}")
        if ui == "streamlit":
            streamlit_process = process_client.start_background_process(
                command=streamlit_command, cwd=root, log_path=streamlit_log_path, env=utf8_env,
            )
            streamlit_created = process_client.get_process_create_time(streamlit_process.pid)
            if not process_client.wait_for_port_open(STREAMLIT_PORT):
                raise RuntimeError(f"Streamlit port({STREAMLIT_PORT}) was not opened. Log: {streamlit_log_path}")

        service_state.save_service_state(service_state.ServiceState(
            ui=ui, fastapi_pid=fastapi_process.pid, fastapi_create_time=fastapi_created,
            streamlit_pid=streamlit_process.pid if streamlit_process else None,
            streamlit_create_time=streamlit_created,
            fastapi_log=str(fastapi_log_path),
            streamlit_log=str(streamlit_log_path) if streamlit_process else None,
            fastapi_port=FASTAPI_PORT, streamlit_port=STREAMLIT_PORT if streamlit_process else None,
            started_at_unix=int(time.time()), platform=sys.platform,
        ))
        log_event(logger, logging.INFO, "services_running", ui=ui,
                  fastapi_url=f"http://127.0.0.1:{FASTAPI_PORT}", ui_url=_ui_url(ui))
        return 0
    except (OSError, RuntimeError) as exc:
        log_event(logger, logging.ERROR, "service_start_failed", error=exc)
        if streamlit_process is not None:
            process_client.terminate_process_tree(streamlit_process.pid, "Streamlit(launcher)", streamlit_created)
        if fastapi_process is not None:
            process_client.terminate_process_tree(fastapi_process.pid, "FastAPI(launcher)", fastapi_created)
        service_state.remove_service_state()
        return 1


def _stop_web_services() -> int:
    state = service_state.load_service_state()
    if state is None:
        log_event(logger, logging.INFO, "service_stop_skipped", reason="no_managed_state")
        return 0
    failed = False
    for name, (pid, created) in _resolve_managed_processes(state).items():
        if pid is not None and not process_client.terminate_process_tree(pid, name, created):
            failed = True
    if not failed:
        service_state.remove_service_state()
    return 1 if failed else 0


def _show_web_status() -> int:
    state = service_state.load_service_state()
    processes = _resolve_managed_processes(state)
    ui = state.ui if state is not None else "react"
    running = processes["FastAPI"][0] is not None and (ui == "react" or processes["Streamlit"][0] is not None)
    log_event(logger, logging.INFO, "web_service_status", ui=ui,
              status="running" if running else "stopped_or_incomplete",
              fastapi_pid=processes["FastAPI"][0], streamlit_pid=processes["Streamlit"][0],
              ui_url=_ui_url(ui) if running else None)
    return 0 if running else 1


def _resolve_managed_processes(state: service_state.ServiceState | None) -> dict[str, tuple[int | None, float | None]]:
    if state is None:
        return {"Streamlit": (None, None), "FastAPI": (None, None)}
    root = get_project_root_path()
    return {
        "Streamlit": process_client.resolve_service_process(
            name="Streamlit", pid=state.streamlit_pid, create_time=state.streamlit_create_time,
            fallback_tokens=STREAMLIT_PROCESS_TOKENS, expected_cwd=root,
        ),
        "FastAPI": process_client.resolve_service_process(
            name="FastAPI", pid=state.fastapi_pid, create_time=state.fastapi_create_time,
            fallback_tokens=FASTAPI_PROCESS_TOKENS, expected_cwd=root,
        ),
    }


def _ui_url(ui: str) -> str:
    return f"http://127.0.0.1:{FASTAPI_PORT}/app/" if ui == "react" else f"http://127.0.0.1:{STREAMLIT_PORT}"
