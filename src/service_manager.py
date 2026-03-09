from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .logging_utils import configure_logging, log_event
from .runtime_encoding import build_utf8_env, ensure_utf8_stdio, maybe_reexec_with_utf8
from .runtime_paths import get_project_root_path, get_runtime_log_path, get_service_state_path
from .settings import ConfigurationError, get_settings, validate_required_keys


ensure_utf8_stdio()
configure_logging()
logger = logging.getLogger(__name__)


STATE_SCHEMA_VERSION = 2
FASTAPI_LOG_FILE = "fastapi.log"
STREAMLIT_LOG_FILE = "streamlit.log"
FASTAPI_PORT = 8000
STREAMLIT_PORT = 8501
CREATE_TIME_TOLERANCE_SEC = 1.0
PROCESS_STOP_TIMEOUT_SEC = 5.0
PROCESS_KILL_TIMEOUT_SEC = 2.0

FASTAPI_PROCESS_TOKENS = ["uvicorn", "src.web.app:app"]
STREAMLIT_PROCESS_TOKENS = ["streamlit", "src/web/streamlit_app.py"]


def _load_validated_settings(context: str):
    settings = get_settings()
    validate_required_keys(settings, context=context)
    return settings


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_service_state() -> dict:
    state_path = get_service_state_path()
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log_event(logger, logging.WARNING, "service_state_load_failed", state_path=state_path, error=exc)
        return {}


def _save_service_state(state: dict) -> None:
    state_path = get_service_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _remove_service_state() -> None:
    state_path = get_service_state_path()
    if state_path.exists():
        state_path.unlink()


def _normalize_cmd_token(value: str) -> str:
    normalized = value.strip().lower().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def _get_process(
    pid: int | None,
    expected_create_time: float | None = None,
) -> psutil.Process | None:
    pid_int = _as_int(pid)
    if pid_int is None or pid_int <= 0:
        return None

    try:
        process = psutil.Process(pid_int)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        return None

    if expected_create_time is None:
        return process

    try:
        actual_create_time = float(process.create_time())
    except (psutil.Error, OSError, ValueError):
        return None

    if abs(actual_create_time - expected_create_time) > CREATE_TIME_TOLERANCE_SEC:
        return None
    return process


def _get_process_create_time(pid: int | None) -> float | None:
    process = _get_process(pid)
    if process is None:
        return None
    try:
        return float(process.create_time())
    except (psutil.Error, OSError, ValueError):
        return None


def _is_process_alive(
    pid: int | None,
    expected_create_time: float | None = None,
) -> bool:
    process = _get_process(pid=pid, expected_create_time=expected_create_time)
    if process is None:
        return False
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, OSError):
        return False


def _is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port_open(
    port: int,
    host: str = "127.0.0.1",
    timeout_sec: float = 20.0,
    interval_sec: float = 0.2,
) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _is_port_open(port=port, host=host):
            return True
        time.sleep(interval_sec)
    return _is_port_open(port=port, host=host)


def _find_process_pid_by_tokens(tokens: list[str]) -> int | None:
    normalized_tokens = [_normalize_cmd_token(token) for token in tokens if token]
    if not normalized_tokens:
        return None

    matches: list[tuple[float, int]] = []
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            cmdline = process.info.get("cmdline") or []
            normalized_cmdline = [_normalize_cmd_token(str(part)) for part in cmdline if part]
            if not normalized_cmdline:
                continue
            if not all(
                any(token in cmd_part for cmd_part in normalized_cmdline)
                for token in normalized_tokens
            ):
                continue

            create_time = float(process.info.get("create_time") or 0.0)
            matches.append((create_time, process.pid))
        except (psutil.Error, OSError, ValueError, TypeError):
            continue

    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def _start_background_process(
    command: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    time.sleep(0.8)
    if process.poll() is not None:
        raise RuntimeError(f"Process start failed: {' '.join(command)} (log: {log_path})")
    return process


def _terminate_process_tree(
    pid: int | None,
    name: str,
    expected_create_time: float | None = None,
) -> bool:
    if pid is None:
        log_event(logger, logging.INFO, "service_stop_skipped", service=name, reason="missing_pid")
        return False

    process = _get_process(pid=pid, expected_create_time=expected_create_time)
    if process is None:
        log_event(logger, logging.INFO, "service_already_stopped", service=name, pid=pid)
        return False

    try:
        descendants = process.children(recursive=True)
    except (psutil.Error, OSError):
        descendants = []

    targets = descendants + [process]
    for target in targets:
        try:
            target.terminate()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (psutil.AccessDenied, psutil.Error, OSError):
            continue

    _, alive = psutil.wait_procs(targets, timeout=PROCESS_STOP_TIMEOUT_SEC)
    if alive:
        for target in alive:
            try:
                target.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.AccessDenied, psutil.Error, OSError):
                continue
        _, alive = psutil.wait_procs(alive, timeout=PROCESS_KILL_TIMEOUT_SEC)

    if alive:
        log_event(logger, logging.ERROR, "service_stop_failed", service=name, pid=pid)
        return False

    log_event(logger, logging.INFO, "service_stopped", service=name, pid=pid)
    return True


def _resolve_service_process(
    name: str,
    pid: int | None,
    create_time: float | None,
    fallback_tokens: list[str],
) -> tuple[int | None, float | None]:
    if _is_process_alive(pid=pid, expected_create_time=create_time):
        return pid, create_time

    if pid is not None:
        log_event(logger, logging.INFO, "service_state_stale", service=name, pid=pid)

    discovered_pid = _find_process_pid_by_tokens(fallback_tokens)
    if discovered_pid is None:
        return None, None

    discovered_create_time = _get_process_create_time(discovered_pid)
    log_event(
        logger,
        logging.INFO,
        "service_discovered_from_cmdline",
        service=name,
        pid=discovered_pid,
    )
    return discovered_pid, discovered_create_time


def _start_web_services() -> int:
    root = get_project_root_path()
    state = _load_service_state()

    fastapi_pid = _as_int(state.get("fastapi_pid"))
    streamlit_pid = _as_int(state.get("streamlit_pid"))
    fastapi_create_time = _as_float(state.get("fastapi_create_time"))
    streamlit_create_time = _as_float(state.get("streamlit_create_time"))

    running = []
    if _is_process_alive(fastapi_pid, fastapi_create_time):
        running.append(f"FastAPI(pid={fastapi_pid})")
    if _is_process_alive(streamlit_pid, streamlit_create_time):
        running.append(f"Streamlit(pid={streamlit_pid})")

    if running:
        log_event(logger, logging.WARNING, "services_already_running", services=", ".join(running))
        return 1

    occupied_ports = []
    if _is_port_open(FASTAPI_PORT):
        occupied_ports.append(f"FastAPI:{FASTAPI_PORT}")
    if _is_port_open(STREAMLIT_PORT):
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
        fastapi_proc = _start_background_process(
            command=fastapi_cmd,
            cwd=root,
            log_path=fastapi_log_path,
            env=utf8_env,
        )
        if not _wait_for_port_open(FASTAPI_PORT):
            raise RuntimeError(f"FastAPI port({FASTAPI_PORT}) was not opened. Log: {fastapi_log_path}")
        fastapi_create_time = _get_process_create_time(fastapi_proc.pid)
        log_event(
            logger,
            logging.INFO,
            "service_started",
            service="FastAPI",
            pid=fastapi_proc.pid,
            log_path=fastapi_log_path,
        )

        streamlit_proc = _start_background_process(
            command=streamlit_cmd,
            cwd=root,
            log_path=streamlit_log_path,
            env=utf8_env,
        )
        if not _wait_for_port_open(STREAMLIT_PORT):
            raise RuntimeError(f"Streamlit port({STREAMLIT_PORT}) was not opened. Log: {streamlit_log_path}")
        streamlit_create_time = _get_process_create_time(streamlit_proc.pid)
        log_event(
            logger,
            logging.INFO,
            "service_started",
            service="Streamlit",
            pid=streamlit_proc.pid,
            log_path=streamlit_log_path,
        )

        _save_service_state(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "fastapi_pid": fastapi_proc.pid,
                "fastapi_create_time": fastapi_create_time,
                "streamlit_pid": streamlit_proc.pid,
                "streamlit_create_time": streamlit_create_time,
                "fastapi_log": str(fastapi_log_path),
                "streamlit_log": str(streamlit_log_path),
                "fastapi_port": FASTAPI_PORT,
                "streamlit_port": STREAMLIT_PORT,
                "started_at_unix": int(time.time()),
                "platform": sys.platform,
            }
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
            _terminate_process_tree(streamlit_proc.pid, "Streamlit(launcher)")
        if fastapi_proc:
            _terminate_process_tree(fastapi_proc.pid, "FastAPI(launcher)")
        _remove_service_state()
        return 1


def _stop_web_services() -> int:
    state = _load_service_state()
    if not state:
        log_event(logger, logging.INFO, "service_state_missing", state_path=get_service_state_path())

    streamlit_pid, streamlit_create_time = _resolve_service_process(
        name="Streamlit",
        pid=_as_int(state.get("streamlit_pid")),
        create_time=_as_float(state.get("streamlit_create_time")),
        fallback_tokens=STREAMLIT_PROCESS_TOKENS,
    )
    fastapi_pid, fastapi_create_time = _resolve_service_process(
        name="FastAPI",
        pid=_as_int(state.get("fastapi_pid")),
        create_time=_as_float(state.get("fastapi_create_time")),
        fallback_tokens=FASTAPI_PROCESS_TOKENS,
    )

    if not streamlit_pid and not fastapi_pid:
        if state:
            _remove_service_state()
        log_event(logger, logging.INFO, "service_stop_skipped", reason="no_running_services")
        return 0

    streamlit_stopped = (
        _terminate_process_tree(
            pid=streamlit_pid,
            name="Streamlit",
            expected_create_time=streamlit_create_time,
        )
        if streamlit_pid is not None
        else False
    )
    fastapi_stopped = (
        _terminate_process_tree(
            pid=fastapi_pid,
            name="FastAPI",
            expected_create_time=fastapi_create_time,
        )
        if fastapi_pid is not None
        else False
    )

    fastapi_remaining = _find_process_pid_by_tokens(FASTAPI_PROCESS_TOKENS)
    streamlit_remaining = _find_process_pid_by_tokens(STREAMLIT_PROCESS_TOKENS)
    if state and fastapi_remaining is None and streamlit_remaining is None:
        _remove_service_state()

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["startweb", "stopweb"],
        help="Service mode: startweb or stopweb.",
    )
    return parser


def main() -> int:
    maybe_reexec_with_utf8("src.service_manager", sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args()
    return run_web_service(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
