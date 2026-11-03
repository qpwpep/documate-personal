"""The private worker requires its runner's live process identity before CLI handling."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest


@pytest.mark.parametrize(
    ("owner", "expected_status"),
    [("missing", 1), ("current", 0), ("different_creation_time", 1)],
)
def test_worker_requires_live_owner_before_processing_arguments(owner: str, expected_status: int) -> None:
    environment = os.environ.copy()
    environment.pop("DOCUMATE_CONVERSION_OWNER_PID", None)
    environment.pop("DOCUMATE_CONVERSION_OWNER_CREATED", None)
    if owner != "missing":
        created = psutil.Process().create_time()
        environment["DOCUMATE_CONVERSION_OWNER_PID"] = str(os.getpid())
        environment["DOCUMATE_CONVERSION_OWNER_CREATED"] = str(
            created + 1 if owner == "different_creation_time" else created
        )

    completed = subprocess.run(
        [sys.executable, "-m", "src.infra.docling_worker", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    assert completed.returncode == expected_status, completed.stderr
    if expected_status == 0:
        assert "--request" in completed.stdout
    else:
        assert completed.stdout == ""
