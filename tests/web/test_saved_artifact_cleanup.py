from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.app.web.cleanup import RuntimeCleaner
from src.core.save_contract import SaveManifest, SaveOperation
from src.infra.saved_artifacts import ArtifactError, manifest_path_for, save_artifact
from src.infra.settings import AppSettings


@pytest.fixture
def cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr("src.app.web.cleanup.get_save_text_output_dir", lambda: tmp_path)
    cleaner = RuntimeCleaner(
        settings=AppSettings(_env_file=None),
        session_store=SimpleNamespace(active_session_ids=lambda: set()),
    )
    return cleaner, tmp_path


def _managed_file(root: Path, *, expires_at: float):
    """Exercise retention against files published by the production writer."""
    text = "보존할 답변"
    operation = SaveOperation.for_text(
        text, operation_id="save-cleanup", session_id="session-cleanup", request_id="request-cleanup",
        contract_revision=1, target_kind="compose", source_hash="a" * 64, answer_hash="b" * 64,
    )
    with patch("src.infra.saved_artifacts.time", return_value=1.0):
        _, payload = save_artifact(root, text, operation, ttl_seconds=int(expires_at - 1))
    sidecar = manifest_path_for(payload)
    return payload, sidecar


def _stats(*, scanned=0, deleted=0, errors=0, staging_scanned=0, staging_deleted=0, busy=0):
    return dict(scanned=scanned, deleted=deleted, errors=errors, staging_scanned=staging_scanned,
                staging_deleted=staging_deleted, busy=busy)


def test_managed_retention_does_not_expire_from_payload_mtime(cleanup):
    cleaner, root = cleanup
    payload, manifest = _managed_file(root, expires_at=200)
    original = payload.read_bytes()
    os.utime(payload, (0, 0))

    result = cleaner.cleanup_expired_generated_files(now_epoch=100, ttl_seconds=10)

    assert result == _stats(scanned=1)
    assert payload.read_bytes() == original
    assert manifest.exists()


def test_expiry_removes_payload_and_retains_idempotency_tombstone(cleanup):
    cleaner, root = cleanup
    payload, manifest = _managed_file(root, expires_at=90)
    original_manifest = manifest.read_bytes()

    result = cleaner.cleanup_expired_generated_files(now_epoch=100, ttl_seconds=10000)

    assert result == _stats(scanned=1, deleted=1)
    assert not payload.exists()
    assert manifest.read_bytes() == original_manifest
    operation = SaveManifest.model_validate_json(original_manifest).operation
    with pytest.raises(ArtifactError) as error:
        save_artifact(root, "보존할 답변", operation, ttl_seconds=10000)
    assert error.value.code == "artifact_expired"
    assert not payload.exists()












def test_cleanup_skips_an_active_writer_even_when_staging_mtime_is_old(cleanup, monkeypatch):
    cleaner, root = cleanup
    ready, resume = Event(), Event()
    real_link = os.link

    def pause_payload_publication(source, destination):
        if Path(destination).suffix == ".txt":
            for staged in root.glob(".save-*.part"):
                os.utime(staged, (0, 0))
            ready.set()
            assert resume.wait(timeout=20)
        return real_link(source, destination)

    monkeypatch.setattr(os, "link", pause_payload_publication)
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(_managed_file, root, expires_at=200)
        try:
            assert ready.wait(timeout=20)
            assert cleaner.cleanup_expired_generated_files(now_epoch=100, ttl_seconds=10) == _stats(busy=1)
            assert len(list(root.glob(".save-*.part"))) == 2
        finally:
            resume.set()
        payload, sidecar = writer.result(timeout=20)
    assert payload.read_text(encoding="utf-8-sig") == "보존할 답변"
    assert sidecar.exists()
    assert list(root.glob(".save-*.part")) == []


_INTERRUPTED_WRITER = r"""
import os
from pathlib import Path
import sys
from src.core.save_contract import SaveOperation
from src.infra import saved_artifacts

phase = sys.argv[2]
def interrupt():
    print('staged', flush=True)
    sys.stdin.buffer.read(1)
    # Exit the real interpreter, including when sys.executable is a Windows
    # venv launcher. No Python finally blocks or atexit handlers may run.
    os._exit(73)
real_write, real_link = os.write, os.link
def interrupted_write(descriptor, payload):
    real_write(descriptor, payload[:3])
    interrupt()
def interrupted_link(source, destination):
    real_link(source, destination)
    if (phase == 'manifest' and Path(destination).suffix == '.json'
            or phase == 'payload' and Path(destination).suffix == '.txt'):
        interrupt()
if phase == 'partial':
    os.write = interrupted_write
else:
    os.link = interrupted_link
saved_artifacts.time = lambda: 100.0
text = 'interrupted answer'
operation = SaveOperation.for_text(
    text, operation_id='killed-save', session_id='killed-session', request_id='request',
    contract_revision=1, target_kind='compose', source_hash='a' * 64, answer_hash='b' * 64,
)
saved_artifacts.save_artifact(Path(sys.argv[1]), text, operation, ttl_seconds=60)
"""


