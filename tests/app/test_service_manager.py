"""Recorded process identity includes the owning project and command."""
from __future__ import annotations

import sys
from pathlib import Path

from src.app.service_manager import process_client


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
