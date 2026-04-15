from __future__ import annotations

import logging
import socket
import subprocess
import time
from pathlib import Path

import psutil

from ..logging_utils import log_event
from ._bootstrap import logger


CREATE_TIME_TOLERANCE_SEC = 1.0
PROCESS_STOP_TIMEOUT_SEC = 5.0
PROCESS_KILL_TIMEOUT_SEC = 2.0


def as_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: object) -> float | None:
    try:
        return float(value)
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


def is_process_alive(
    pid: int | None,
    expected_create_time: float | None = None,
) -> bool:
    process = get_process(pid=pid, expected_create_time=expected_create_time)
    if process is None:
        return False
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, OSError):
        return False


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


def find_process_pid_by_tokens(tokens: list[str]) -> int | None:
    normalized_tokens = [normalize_cmd_token(token) for token in tokens if token]
    if not normalized_tokens:
        return None

    matches: list[tuple[float, int]] = []
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            cmdline = process.info.get("cmdline") or []
            normalized_cmdline = [normalize_cmd_token(str(part)) for part in cmdline if part]
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


def start_background_process(
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
    fallback_tokens: list[str],
) -> tuple[int | None, float | None]:
    if is_process_alive(pid=pid, expected_create_time=create_time):
        return pid, create_time

    if pid is not None:
        log_event(logger, logging.INFO, "service_state_stale", service=name, pid=pid)

    discovered_pid = find_process_pid_by_tokens(fallback_tokens)
    if discovered_pid is None:
        return None, None

    discovered_create_time = get_process_create_time(discovered_pid)
    log_event(
        logger,
        logging.INFO,
        "service_discovered_from_cmdline",
        service=name,
        pid=discovered_pid,
    )
    return discovered_pid, discovered_create_time
