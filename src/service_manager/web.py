from __future__ import annotations

import logging
import os
import sys
import time

from ..logging_utils import log_event
from ..runtime_encoding import build_utf8_env
from ..runtime_paths import get_project_root_path, get_runtime_log_path
from ..settings import ConfigurationError, get_settings, validate_required_keys
from ._bootstrap import logger
from . import process_client
from . import state as service_state


FASTAPI_LOG_FILE = "fastapi.log"
STREAMLIT_LOG_FILE = "streamlit.log"
FASTAPI_PORT = 8000
STREAMLIT_PORT = 8501

FASTAPI_PROCESS_TOKENS = ["uvicorn", "src.web.app:app"]
STREAMLIT_PROCESS_TOKENS = ["streamlit", "src/web/streamlit_app.py"]


def _load_validated_settings(context: str):
    settings = get_settings()
    validate_required_keys(settings, context=context)
    return settings


def _start_web_services() -> int:
    root = get_project_root_path()
    state = service_state.load_service_state()

    fastapi_pid = state.fastapi_pid if state is not None else None
    streamlit_pid = state.streamlit_pid if state is not None else None
    fastapi_create_time = state.fastapi_create_time if state is not None else None
    streamlit_create_time = state.streamlit_create_time if state is not None else None

    running = []
    if process_client.is_process_alive(fastapi_pid, fastapi_create_time):
        running.append(f"FastAPI(pid={fastapi_pid})")
    if process_client.is_process_alive(streamlit_pid, streamlit_create_time):
        running.append(f"Streamlit(pid={streamlit_pid})")

    if running:
        log_event(logger, logging.WARNING, "services_already_running", services=", ".join(running))
        return 1

    occupied_ports = []
    if process_client.is_port_open(FASTAPI_PORT):
        occupied_ports.append(f"FastAPI:{FASTAPI_PORT}")
    if process_client.is_port_open(STREAMLIT_PORT):
        occupied_ports.append(f"Streamlit:{STREAMLIT_PORT}")
    if occupied_ports:
        log_event(logger, logging.ERROR, "ports_unavailable", ports=", ".join(occupied_ports))
        return 1

    fastapi_log_path = get_runtime_log_path(FASTAPI_LOG_FILE)
    streamlit_log_path = get_runtime_log_path(STREAMLIT_LOG_FILE)

    utf8_env = build_utf8_env(os.environ.copy())
    fastapi_cmd = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "uvicorn",
        "src.web.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(FASTAPI_PORT),
    ]
    streamlit_cmd = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "streamlit",
        "run",
        "src/web/streamlit_app.py",
        "--server.port",
        str(STREAMLIT_PORT),
    ]

    fastapi_proc = None
    streamlit_proc = None
    try:
        fastapi_proc = process_client.start_background_process(
            command=fastapi_cmd,
            cwd=root,
            log_path=fastapi_log_path,
            env=utf8_env,
        )
        if not process_client.wait_for_port_open(FASTAPI_PORT):
            raise RuntimeError(f"FastAPI port({FASTAPI_PORT}) was not opened. Log: {fastapi_log_path}")
        fastapi_create_time = process_client.get_process_create_time(fastapi_proc.pid)
        log_event(
            logger,
            logging.INFO,
            "service_started",
            service="FastAPI",
            pid=fastapi_proc.pid,
            log_path=fastapi_log_path,
        )

        streamlit_proc = process_client.start_background_process(
            command=streamlit_cmd,
            cwd=root,
            log_path=streamlit_log_path,
            env=utf8_env,
        )
        if not process_client.wait_for_port_open(STREAMLIT_PORT):
            raise RuntimeError(f"Streamlit port({STREAMLIT_PORT}) was not opened. Log: {streamlit_log_path}")
        streamlit_create_time = process_client.get_process_create_time(streamlit_proc.pid)
        log_event(
            logger,
            logging.INFO,
            "service_started",
            service="Streamlit",
            pid=streamlit_proc.pid,
            log_path=streamlit_log_path,
        )

        service_state.save_service_state(
            service_state.ServiceState(
                fastapi_pid=fastapi_proc.pid,
                fastapi_create_time=fastapi_create_time,
                streamlit_pid=streamlit_proc.pid,
                streamlit_create_time=streamlit_create_time,
                fastapi_log=str(fastapi_log_path),
                streamlit_log=str(streamlit_log_path),
                fastapi_port=FASTAPI_PORT,
                streamlit_port=STREAMLIT_PORT,
                started_at_unix=int(time.time()),
                platform=sys.platform,
            )
        )
        log_event(
            logger,
            logging.INFO,
            "services_running",
            fastapi_url=f"http://localhost:{FASTAPI_PORT}",
            streamlit_url=f"http://localhost:{STREAMLIT_PORT}",
        )
        return 0
    except Exception as exc:
        log_event(logger, logging.ERROR, "service_start_failed", error=exc)
        if streamlit_proc:
            process_client.terminate_process_tree(streamlit_proc.pid, "Streamlit(launcher)")
        if fastapi_proc:
            process_client.terminate_process_tree(fastapi_proc.pid, "FastAPI(launcher)")
        service_state.remove_service_state()
        return 1


def _stop_web_services() -> int:
    state = service_state.load_service_state()
    if state is None:
        log_event(logger, logging.INFO, "service_state_missing", state_path=service_state.get_service_state_path())

    streamlit_pid, streamlit_create_time = process_client.resolve_service_process(
        name="Streamlit",
        pid=process_client.as_int(state.streamlit_pid) if state is not None else None,
        create_time=process_client.as_float(state.streamlit_create_time) if state is not None else None,
        fallback_tokens=STREAMLIT_PROCESS_TOKENS,
    )
    fastapi_pid, fastapi_create_time = process_client.resolve_service_process(
        name="FastAPI",
        pid=process_client.as_int(state.fastapi_pid) if state is not None else None,
        create_time=process_client.as_float(state.fastapi_create_time) if state is not None else None,
        fallback_tokens=FASTAPI_PROCESS_TOKENS,
    )

    if not streamlit_pid and not fastapi_pid:
        if state is not None:
            service_state.remove_service_state()
        log_event(logger, logging.INFO, "service_stop_skipped", reason="no_running_services")
        return 0

    streamlit_stopped = (
        process_client.terminate_process_tree(
            pid=streamlit_pid,
            name="Streamlit",
            expected_create_time=streamlit_create_time,
        )
        if streamlit_pid is not None
        else False
    )
    fastapi_stopped = (
        process_client.terminate_process_tree(
            pid=fastapi_pid,
            name="FastAPI",
            expected_create_time=fastapi_create_time,
        )
        if fastapi_pid is not None
        else False
    )

    fastapi_remaining = process_client.find_process_pid_by_tokens(FASTAPI_PROCESS_TOKENS)
    streamlit_remaining = process_client.find_process_pid_by_tokens(STREAMLIT_PROCESS_TOKENS)
    if state is not None and fastapi_remaining is None and streamlit_remaining is None:
        service_state.remove_service_state()

    log_event(
        logger,
        logging.INFO,
        "service_stop_complete",
        fastapi_stopped=fastapi_stopped,
        streamlit_stopped=streamlit_stopped,
    )
    return 0


def run_web_service(mode: str) -> int:
    if mode == "startweb":
        try:
            _load_validated_settings("startweb")
        except ConfigurationError as exc:
            log_event(logger, logging.ERROR, "configuration_error", context="startweb", error=exc)
            return 1
        return _start_web_services()
    if mode == "stopweb":
        return _stop_web_services()

    log_event(logger, logging.ERROR, "unsupported_mode", mode=mode)
    return 1
