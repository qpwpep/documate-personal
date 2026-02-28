import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessage

from .settings import ConfigurationError, get_settings, validate_required_keys
from .util.util import get_project_root_path


SERVICE_STATE_FILE = "script/web_services_state.json"
FASTAPI_LOG_FILE = "fastapi.log"
STREAMLIT_LOG_FILE = "streamlit.log"
FASTAPI_PORT = 8000
STREAMLIT_PORT = 8501


def maybe_save_mermaid_png(graph):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Saved graph to graph.png")
    except Exception:
        pass


def _load_validated_settings(context: str):
    settings = get_settings()
    validate_required_keys(settings, context=context)
    return settings


def run_cli():
    try:
        settings = _load_validated_settings("cli")
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return

    from .graph_builder import build_agent_graph

    graph = build_agent_graph(settings=settings)
    maybe_save_mermaid_png(graph)
    verbose = settings.verbose

    messages = []
    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            state = {
                "user_input": user_input,
                "messages": messages,
            }
            response = graph.invoke(state)
            messages = response["messages"]

            if verbose:
                for msg in response["messages"]:
                    try:
                        msg.pretty_print()
                    except Exception:
                        print(repr(msg))
            else:
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        print(msg.content)
                        break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as exc:
            print("Error:", exc)
            break


def _service_state_path(root_path: str) -> Path:
    return Path(root_path) / SERVICE_STATE_FILE


def _load_service_state(root_path: str) -> dict:
    state_path = _service_state_path(root_path)
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_service_state(root_path: str, state: dict) -> None:
    state_path = _service_state_path(root_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _remove_service_state(root_path: str) -> None:
    state_path = _service_state_path(root_path)
    if state_path.exists():
        state_path.unlink()


def _is_process_alive(pid: int | None) -> bool:
    if not pid:
        return False

    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip()
        if not output or output.startswith("INFO:"):
            return False
        return str(pid) in output

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _start_background_process(command: list[str], cwd: str, log_path: str) -> subprocess.Popen:
    with open(log_path, "a", encoding="utf-8") as log_file:
        kwargs = {
            "cwd": cwd,
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            detached = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            process = subprocess.Popen(command, creationflags=detached, **kwargs)
        else:
            process = subprocess.Popen(command, start_new_session=True, **kwargs)

    time.sleep(0.8)
    if process.poll() is not None:
        raise RuntimeError(f"Process start failed: {' '.join(command)} (log: {log_path})")
    return process


def _wait_until_stopped(pid: int, timeout_sec: float = 5.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not _is_process_alive(pid):
            return True
        time.sleep(0.1)
    return not _is_process_alive(pid)


def _get_listening_pids(port: int) -> set[int]:
    pids: set[int] = set()

    if os.name == "nt":
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            if parts[3].upper() != "LISTENING":
                continue
            if not parts[1].endswith(f":{port}"):
                continue
            try:
                pids.add(int(parts[4]))
            except ValueError:
                continue
        return pids

    if shutil.which("lsof"):
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            try:
                pids.add(int(line.strip()))
            except ValueError:
                continue
        return pids

    if shutil.which("ss"):
        result = subprocess.run(
            ["ss", "-ltnp"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            if f":{port} " not in line:
                continue
            match = re.search(r"pid=(\d+)", line)
            if match:
                pids.add(int(match.group(1)))
        return pids

    return pids


def _wait_for_service_pid(port: int, before_pids: set[int], timeout_sec: float = 20.0) -> int | None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        current_pids = _get_listening_pids(port)
        if current_pids:
            new_pids = current_pids - before_pids
            target = sorted(new_pids or current_pids)
            if target:
                return target[0]
        time.sleep(0.2)
    return None


def _terminate_process(pid: int | None, name: str) -> bool:
    if not pid:
        print(f"- {name}: PID information is missing.")
        return False

    if not _is_process_alive(pid):
        print(f"- {name}: already stopped (PID {pid})")
        return False

    try:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                print(f"- {name}: stopped (PID {pid})")
                return True
            print(f"- {name}: stop failed (PID {pid})")
            if result.stdout:
                print(result.stdout.strip())
            if result.stderr:
                print(result.stderr.strip())
            return False

        os.killpg(pid, signal.SIGTERM)
        if _wait_until_stopped(pid, timeout_sec=5):
            print(f"- {name}: stopped (PID {pid})")
            return True
        os.killpg(pid, signal.SIGKILL)
        stopped = _wait_until_stopped(pid, timeout_sec=2)
        print(f"- {name}: stopped (PID {pid})" if stopped else f"- {name}: stop failed (PID {pid})")
        return stopped
    except ProcessLookupError:
        print(f"- {name}: already stopped (PID {pid})")
        return False
    except Exception as exc:
        print(f"- {name}: error while stopping (PID {pid}) -> {exc}")
        return False


def _start_web_services(root_path: str) -> None:
    state = _load_service_state(root_path)

    fastapi_pid = state.get("fastapi_pid")
    streamlit_pid = state.get("streamlit_pid")
    running = []
    if _is_process_alive(fastapi_pid):
        running.append(f"FastAPI(pid={fastapi_pid})")
    if _is_process_alive(streamlit_pid):
        running.append(f"Streamlit(pid={streamlit_pid})")

    if running:
        print("Services are already running.")
        print(" / ".join(running))
        print("Run 'uv run python -m src.main --mode stopweb' first.")
        return

    fastapi_log_path = os.path.join(root_path, FASTAPI_LOG_FILE)
    streamlit_log_path = os.path.join(root_path, STREAMLIT_LOG_FILE)

    fastapi_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.web.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(FASTAPI_PORT),
    ]
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/web/streamlit_app.py",
        "--server.port",
        str(STREAMLIT_PORT),
    ]

    print("=" * 60)
    print("Starting web services...")
    print("=" * 60)

    fastapi_proc = None
    streamlit_proc = None
    fastapi_service_pid = None
    streamlit_service_pid = None
    try:
        fastapi_before = _get_listening_pids(FASTAPI_PORT)
        fastapi_proc = _start_background_process(
            command=fastapi_cmd,
            cwd=root_path,
            log_path=fastapi_log_path,
        )
        fastapi_service_pid = _wait_for_service_pid(FASTAPI_PORT, fastapi_before)
        if fastapi_service_pid is None:
            raise RuntimeError(f"FastAPI port({FASTAPI_PORT}) was not opened. Log: {fastapi_log_path}")
        print(
            f"- FastAPI started (launcher PID {fastapi_proc.pid}, service PID {fastapi_service_pid}, log: {fastapi_log_path})"
        )

        streamlit_before = _get_listening_pids(STREAMLIT_PORT)
        streamlit_proc = _start_background_process(
            command=streamlit_cmd,
            cwd=root_path,
            log_path=streamlit_log_path,
        )
        streamlit_service_pid = _wait_for_service_pid(STREAMLIT_PORT, streamlit_before)
        if streamlit_service_pid is None:
            raise RuntimeError(f"Streamlit port({STREAMLIT_PORT}) was not opened. Log: {streamlit_log_path}")
        print(
            f"- Streamlit started (launcher PID {streamlit_proc.pid}, service PID {streamlit_service_pid}, log: {streamlit_log_path})"
        )

        _save_service_state(
            root_path,
            {
                "fastapi_pid": fastapi_service_pid,
                "streamlit_pid": streamlit_service_pid,
                "fastapi_log": fastapi_log_path,
                "streamlit_log": streamlit_log_path,
                "fastapi_port": FASTAPI_PORT,
                "streamlit_port": STREAMLIT_PORT,
                "started_at_unix": int(time.time()),
                "platform": os.name,
            },
        )

        print("=" * 60)
        print("Web services are running.")
        print("- FastAPI:   http://localhost:8000")
        print("- Streamlit: http://localhost:8501")
        print("- Stop: uv run python -m src.main --mode stopweb")
        print("=" * 60)
    except Exception as exc:
        print(f"Error while starting services: {exc}")
        if streamlit_service_pid:
            _terminate_process(streamlit_service_pid, "Streamlit")
        elif streamlit_proc and _is_process_alive(streamlit_proc.pid):
            _terminate_process(streamlit_proc.pid, "Streamlit(launcher)")
        if fastapi_service_pid:
            _terminate_process(fastapi_service_pid, "FastAPI")
        elif fastapi_proc and _is_process_alive(fastapi_proc.pid):
            _terminate_process(fastapi_proc.pid, "FastAPI(launcher)")
        _remove_service_state(root_path)


def _stop_web_services(root_path: str) -> None:
    state = _load_service_state(root_path)

    if not state:
        print("No saved service state file found. Falling back to port-based termination.")
        streamlit_candidates = sorted(_get_listening_pids(STREAMLIT_PORT))
        fastapi_candidates = sorted(_get_listening_pids(FASTAPI_PORT))

        streamlit_pid = streamlit_candidates[0] if streamlit_candidates else None
        fastapi_pid = fastapi_candidates[0] if fastapi_candidates else None

        if not streamlit_pid and not fastapi_pid:
            print("No running target services found.")
            return

        print("=" * 60)
        print("Stopping web services... (fallback)")
        print("=" * 60)
        streamlit_stopped = _terminate_process(streamlit_pid, "Streamlit")
        fastapi_stopped = _terminate_process(fastapi_pid, "FastAPI")
        print("=" * 60)
        if streamlit_stopped or fastapi_stopped:
            print("Web services stopped.")
        else:
            print("No active process was stopped.")
        print("=" * 60)
        return

    print("=" * 60)
    print("Stopping web services...")
    print("=" * 60)

    streamlit_pid = state.get("streamlit_pid")
    fastapi_pid = state.get("fastapi_pid")

    if not _is_process_alive(streamlit_pid):
        port_pids = _get_listening_pids(state.get("streamlit_port", STREAMLIT_PORT))
        if port_pids:
            streamlit_pid = sorted(port_pids)[0]

    if not _is_process_alive(fastapi_pid):
        port_pids = _get_listening_pids(state.get("fastapi_port", FASTAPI_PORT))
        if port_pids:
            fastapi_pid = sorted(port_pids)[0]

    streamlit_stopped = _terminate_process(streamlit_pid, "Streamlit")
    fastapi_stopped = _terminate_process(fastapi_pid, "FastAPI")

    if not _is_process_alive(streamlit_pid) and not _is_process_alive(fastapi_pid):
        _remove_service_state(root_path)

    print("=" * 60)
    if streamlit_stopped or fastapi_stopped:
        print("Web services stopped.")
    else:
        print("No active process was stopped.")
    print("=" * 60)


def run_web_service(mode: str):
    root_path = str(get_project_root_path())
    if mode == "startweb":
        try:
            _load_validated_settings("startweb")
        except ConfigurationError as exc:
            print(f"Configuration error: {exc}")
            return
        _start_web_services(root_path)
    elif mode == "stopweb":
        _stop_web_services(root_path)
    else:
        print("Unsupported mode.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=["cli", "startweb", "stopweb"],
        help="Execution mode: 'cli', 'startweb', or 'stopweb'.",
    )
    args = parser.parse_args()

    if args.mode in {"startweb", "stopweb"}:
        run_web_service(args.mode)
    else:
        run_cli()
