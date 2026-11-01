"""The launcher serves React through its single recorded FastAPI process."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.app.service_manager import cli, process_client, state, web
from src.infra.settings import AppSettings


@pytest.fixture
def services(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    launched: list[list[str]] = []
    running: set[int] = set()
    stopped: list[int] = []
    occupied: set[int] = set()
    build = tmp_path / "frontend" / "dist"
    build.mkdir(parents=True)
    (build / "index.html").write_text("<main>DocuMate</main>", encoding="utf-8")
    monkeypatch.setattr(web, "get_project_root_path", lambda: tmp_path)
    monkeypatch.setattr(web, "get_frontend_dist_path", lambda: build, raising=False)
    monkeypatch.setattr(web, "get_runtime_log_path", lambda name: tmp_path / name)
    monkeypatch.setattr(state, "get_service_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(web, "get_settings", lambda: AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test-key",
    ))

    def start(*, command, cwd, log_path, env):
        assert cwd == tmp_path
        assert env["PYTHONUTF8"] == "1"
        launched.append(command)
        pid = 100 + len(launched)
        running.add(pid)
        return SimpleNamespace(pid=pid)

    def stop(pid, name, expected_create_time=None):
        assert expected_create_time == float(pid)
        stopped.append(pid)
        running.remove(pid)
        return True

    def resolve(*, pid, create_time, **_kwargs):
        return (pid, float(pid)) if pid in running else (None, None)

    monkeypatch.setattr(process_client, "start_background_process", start)
    monkeypatch.setattr(process_client, "terminate_process_tree", stop)
    monkeypatch.setattr(process_client, "resolve_service_process", resolve)
    monkeypatch.setattr(process_client, "get_process_create_time", lambda pid: float(pid))
    monkeypatch.setattr(process_client, "is_port_open", lambda port: port in occupied)
    monkeypatch.setattr(process_client, "wait_for_port_open", lambda port: True)
    monkeypatch.setattr(cli, "maybe_reexec_with_utf8", lambda *args: None)
    return SimpleNamespace(root=tmp_path, build=build, launched=launched, running=running,
                           stopped=stopped, occupied=occupied)


def test_start_uses_one_python_server(services):
    assert cli.main(["startweb"]) == 0
    assert len(services.launched) == 1
    assert services.launched[0] == [
        sys.executable, "-X", "utf8", "-m", "uvicorn", "src.app.web.app:app",
        "--host", "0.0.0.0", "--port", "8000",
    ]
    saved = state.load_service_state()
    assert saved.fastapi_pid in services.running
    assert json.loads((services.root / "state.json").read_text(encoding="utf-8")) == {
        "schema_version": 4,
        "fastapi_pid": 101,
        "fastapi_create_time": 101.0,
        "fastapi_log": str(services.root / "fastapi.log"),
        "fastapi_port": 8000,
        "started_at_unix": saved.started_at_unix,
        "platform": sys.platform,
    }
    assert cli.main(["statusweb"]) == 0


@pytest.mark.parametrize("mode", ["startweb", "restartweb", "stopweb", "statusweb"])
def test_ui_selector_is_not_supported(services, mode):
    with pytest.raises(SystemExit) as error:
        cli.main([mode, "--ui", "react"])
    assert error.value.code == 2
    assert services.launched == []
    assert services.stopped == []


def test_missing_react_build_explains_build_without_starting_or_stopping(services, caplog):
    assert cli.main(["startweb"]) == 0
    (services.build / "index.html").unlink()
    with caplog.at_level("ERROR"):
        assert cli.main(["restartweb"]) == 1
    assert services.stopped == []
    assert services.running == {101}
    assert "npm --prefix frontend ci" in caplog.text
    assert "npm --prefix frontend run build" in caplog.text


def test_restart_replaces_the_recorded_server(services):
    assert cli.main(["startweb"]) == 0
    services.running.add(999)
    assert cli.main(["restartweb"]) == 0
    assert services.stopped == [101]
    assert len(services.launched) == 2
    assert state.load_service_state().fastapi_pid == 102
    assert services.running == {102, 999}
    assert cli.main(["statusweb"]) == 0


def test_incompatible_state_is_not_adopted(services):
    (services.root / "state.json").write_text(json.dumps({
        "schema_version": 2, "fastapi_pid": 101, "fastapi_create_time": 101.0,
    }), encoding="utf-8")
    services.running.add(101)
    assert cli.main(["statusweb"]) == 1
    assert cli.main(["stopweb"]) == 0
    assert services.stopped == []


def test_missing_state_does_not_discover_or_stop_a_user_process(services):
    services.running.add(999)
    assert cli.main(["stopweb"]) == 0
    assert services.running == {999}
    assert services.stopped == []


def test_stop_does_not_adopt_an_unrecorded_process(services):
    assert cli.main(["startweb"]) == 0
    services.running.add(999)
    assert cli.main(["stopweb"]) == 0
    assert services.stopped == [101]
    assert services.running == {999}
    assert state.load_service_state() is None
    assert cli.main(["statusweb"]) == 1


def test_failed_stop_preserves_state_and_prevents_restart(services, monkeypatch):
    assert cli.main(["startweb"]) == 0
    monkeypatch.setattr(process_client, "terminate_process_tree", lambda *args: False)
    assert cli.main(["restartweb"]) == 1
    assert len(services.launched) == 1
    assert state.load_service_state().fastapi_pid == 101


def test_busy_api_port_does_not_launch_or_stop_anything(services):
    services.occupied.add(8000)
    assert cli.main(["startweb"]) == 1
    assert services.launched == []
    assert services.stopped == []


def test_server_start_failure_cleans_up_its_process_and_state(services, monkeypatch):
    monkeypatch.setattr(process_client, "wait_for_port_open", lambda port: False)
    assert cli.main(["startweb"]) == 1
    assert services.running == set()
    assert services.stopped == [101]
    assert state.load_service_state() is None


def test_recorded_process_identity_includes_project_and_command(tmp_path: Path):
    command = [sys.executable, "-c", "import time; time.sleep(60)", "documate-owned-fixture"]
    process = process_client.start_background_process(command, tmp_path, tmp_path / "fixture.log")
    try:
        created = process_client.get_process_create_time(process.pid)
        arguments = {"name": "fixture", "pid": process.pid, "create_time": created,
                     "expected_tokens": ["documate-owned-fixture"], "expected_cwd": tmp_path}
        assert process_client.resolve_service_process(**arguments) == (process.pid, created)
        assert process_client.resolve_service_process(**{**arguments, "create_time": None}) == (None, None)
        assert process_client.resolve_service_process(**{**arguments, "pid": None}) == (None, None)
        assert process_client.resolve_service_process(**{**arguments, "expected_cwd": tmp_path / "elsewhere"}) == (None, None)
        assert process_client.resolve_service_process(**{**arguments, "expected_tokens": ["other-command"]}) == (None, None)
        assert process_client.resolve_service_process(**{**arguments, "create_time": created - 10}) == (None, None)
        assert process.poll() is None
        assert process_client.terminate_process_tree(process.pid, "fixture", created)
        process.wait(timeout=5)
        assert process_client.get_process(process.pid, created) is None
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
