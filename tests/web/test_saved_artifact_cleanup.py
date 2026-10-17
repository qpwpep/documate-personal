from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from src.app.agent_manager import AgentFlowManager
from src.app.web.cleanup import RuntimeCleaner
from src.app.web.session_store import InMemorySessionStore
from src.core.save_contract import SaveOperation
from src.infra.saved_artifacts import ArtifactError, manifest_path_for, read_saved_artifact, save_artifact
from src.infra.settings import AppSettings


@pytest.fixture
def cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr("src.app.web.cleanup.get_save_text_output_dir", lambda: tmp_path)
    monkeypatch.setattr("src.infra.saved_artifacts.time", lambda: 100.0)
    settings = AppSettings(_env_file=None)
    sessions = InMemorySessionStore(settings, agent_factory=lambda: AgentFlowManager(settings))
    return RuntimeCleaner(settings, sessions), tmp_path


def _save(root: Path):
    text = "보존할 답변"
    operation = SaveOperation.for_text(
        text, operation_id="save-cleanup", session_id="session-cleanup", request_id="request-cleanup",
        contract_revision=1, target_kind="compose", source_hash="a" * 64, answer_hash="b" * 64,
    )
    return save_artifact(root, text, operation, ttl_seconds=100)


def test_managed_retention_does_not_expire_from_payload_mtime(cleanup):
    cleaner, root = cleanup
    manifest, payload = _save(root)
    os.utime(payload, (0, 0))

    result = cleaner.cleanup_expired_generated_files(now_epoch=199, ttl_seconds=10)

    assert result["deleted"] == result["errors"] == 0
    assert read_saved_artifact(root, payload.name, now_epoch=199) == (
        manifest, "보존할 답변".encode("utf-8-sig"),
    )


def test_expiry_removes_payload_and_retains_idempotency_tombstone(cleanup, monkeypatch):
    cleaner, root = cleanup
    manifest, payload = _save(root)
    sidecar = manifest_path_for(payload)
    original_manifest = sidecar.read_bytes()

    result = cleaner.cleanup_expired_generated_files(now_epoch=200, ttl_seconds=10000)

    assert result["deleted"] == 1
    assert result["errors"] == 0
    assert not payload.exists()
    assert sidecar.read_bytes() == original_manifest
    monkeypatch.setattr("src.infra.saved_artifacts.time", lambda: 200.0)
    with pytest.raises(ArtifactError) as error:
        save_artifact(root, "보존할 답변", manifest.operation, ttl_seconds=10000)
    assert error.value.code == "artifact_expired"
    assert not payload.exists()
    assert sidecar.read_bytes() == original_manifest


def test_cleanup_skips_an_active_writer_even_when_staging_mtime_is_old(cleanup, monkeypatch):
    cleaner, root = cleanup
    ready, resume = Event(), Event()
    real_link = os.link

    def pause_payload_publication(source, destination):
        if Path(destination).suffix == ".txt":
            for staged in root.glob(".save-*.part"):
                os.utime(staged, (0, 0))
            ready.set()
            assert resume.wait(timeout=20), "cleanup did not release the paused writer"
        return real_link(source, destination)

    monkeypatch.setattr(os, "link", pause_payload_publication)
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(_save, root)
        try:
            assert ready.wait(timeout=20), "writer did not reach payload publication"
            staged_bytes = {path: path.read_bytes() for path in root.glob(".save-*.part")}
            assert staged_bytes

            result = cleaner.cleanup_expired_generated_files(now_epoch=150, ttl_seconds=10)

            assert result["busy"] == 1
            assert result["deleted"] == result["staging_deleted"] == result["errors"] == 0
            assert {path: path.read_bytes() for path in root.glob(".save-*.part")} == staged_bytes
        finally:
            resume.set()
        manifest, payload = writer.result(timeout=20)

    assert read_saved_artifact(root, payload.name, now_epoch=150) == (
        manifest, "보존할 답변".encode("utf-8-sig"),
    )
    assert list(root.glob(".save-*.part")) == []
