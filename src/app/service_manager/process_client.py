from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

from src.infra.logging_utils import log_event
from src.app.service_manager._bootstrap import logger


CREATE_TIME_TOLERANCE_SEC = 1.0
PROCESS_STOP_TIMEOUT_SEC = 5.0
PROCESS_KILL_TIMEOUT_SEC = 2.0


def as_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_cmd_token(value: str) -> str:
    normalized = value.strip().lower().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def get_process(
    pid: int | None,
    expected_create_time: float | None = None,
) -> psutil.Process | None:
    pid_int = as_int(pid)
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


def get_process_create_time(pid: int | None) -> float | None:
    process = get_process(pid)
    if process is None:
        return None
    try:
        return float(process.create_time())
    except (psutil.Error, OSError, ValueError):
        return None


def is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port_open(
    port: int,
    host: str = "127.0.0.1",
    timeout_sec: float = 20.0,
    interval_sec: float = 0.2,
) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if is_port_open(port=port, host=host):
            return True
        time.sleep(interval_sec)
    return is_port_open(port=port, host=host)


def start_background_process(
    command: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        _write_startup_marker(log_file=log_file, command=command, cwd=cwd)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            env=env,
        )

    time.sleep(0.8)
    if process.poll() is not None:
        raise RuntimeError(f"Process start failed: {' '.join(command)} (log: {log_path})")
    return process


def _write_startup_marker(*, log_file, command: list[str], cwd: Path) -> None:
    if log_file.tell() > 0:
        log_file.write("\n")
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    rendered_command = subprocess.list2cmdline([str(part) for part in command])
    log_file.write(
        f"=== DOCUMATE SERVICE START {timestamp} cwd={cwd} command={rendered_command} ===\n"
    )
    log_file.flush()


def terminate_process_tree(
    pid: int | None,
    name: str,
    expected_create_time: float | None = None,
) -> bool:
    if pid is None:
        log_event(logger, logging.INFO, "service_stop_skipped", service=name, reason="missing_pid")
        return False

    process = get_process(pid=pid, expected_create_time=expected_create_time)
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


def resolve_service_process(
    name: str,
    pid: int | None,
    create_time: float | None,
    expected_tokens: list[str],
    expected_cwd: Path,
) -> tuple[int | None, float | None]:
    # A matching command elsewhere is not proof that this launcher owns it.
    if create_time is None:
        return None, None
    process = get_process(pid=pid, expected_create_time=create_time)
    if process is None:
        return None, None
    try:
        tokens = [normalize_cmd_token(value) for value in process.cmdline()]
        expected = [normalize_cmd_token(value) for value in expected_tokens]
        matches = all(any(part == token or part.endswith("/" + token) for part in tokens)
                      for token in expected)
        if not matches or Path(process.cwd()).resolve() != expected_cwd.resolve():
            log_event(logger, logging.WARNING, "service_state_unverified", service=name, pid=pid)
            return None, None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None, None
        return process.pid, float(process.create_time())
    except (psutil.Error, OSError, ValueError):
        return None, None
