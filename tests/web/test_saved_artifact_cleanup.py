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


def test_broken_manifest_does_not_become_legacy_mtime_cleanup(cleanup):
    cleaner, root = cleanup
    payload, manifest = _managed_file(root, expires_at=200)
    manifest.write_text("broken manifest", encoding="utf-8")
    os.utime(payload, (0, 0))

    result = cleaner.cleanup_expired_generated_files(now_epoch=100, ttl_seconds=10)

    assert result == _stats(scanned=1, errors=1)
    assert payload.exists()
    assert manifest.exists()


def test_legacy_text_files_keep_the_existing_mtime_retention(cleanup):
    cleaner, root = cleanup
    payload = root / "response_20260101_010101.txt"
    payload.write_text("legacy", encoding="utf-8")
    os.utime(payload, (0, 0))

    result = cleaner.cleanup_expired_generated_files(now_epoch=100, ttl_seconds=10)

    assert result == _stats(scanned=1, deleted=1)
    assert not payload.exists()


@pytest.mark.parametrize("cleanup_at", [100, 200])
def test_cleanup_reclaims_failed_staging_unlinks_without_changing_retention(cleanup, monkeypatch, cleanup_at):
    cleaner, root = cleanup
    real_unlink = Path.unlink

    def fail_staging_unlink(path, *args, **kwargs):
        if path.name.startswith(".save-") and path.suffix == ".part":
            raise PermissionError("staging deletion temporarily unavailable")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(Path, "unlink", fail_staging_unlink)
        payload, sidecar = _managed_file(root, expires_at=200)
    original_payload = payload.read_bytes()
    original_manifest = sidecar.read_bytes()
    assert len(list(root.glob(".save-*.part"))) == 2

    result = cleaner.cleanup_expired_generated_files(now_epoch=cleanup_at, ttl_seconds=10)

    assert list(root.glob(".save-*.part")) == []
    assert result == _stats(scanned=1, deleted=int(cleanup_at == 200), staging_scanned=2, staging_deleted=2)
    assert sidecar.read_bytes() == original_manifest
    operation = SaveManifest.model_validate_json(original_manifest).operation
    with patch("src.infra.saved_artifacts.time", return_value=cleanup_at):
        if cleanup_at < 200:
            assert payload.read_bytes() == original_payload
            manifest, retried = save_artifact(root, "보존할 답변", operation, ttl_seconds=10000)
            assert retried == payload
            assert manifest.artifact.expires_at == 200
        else:
            assert not payload.exists()
            with pytest.raises(ArtifactError) as error:
                save_artifact(root, "보존할 답변", operation, ttl_seconds=10000)
            assert error.value.code == "artifact_expired"


def test_cleanup_reclaims_partial_write_even_when_local_error_cleanup_failed(cleanup, monkeypatch):
    cleaner, root = cleanup
    real_write, real_unlink = os.write, Path.unlink

    def fail_write(descriptor, payload):
        real_write(descriptor, payload[:3])
        raise OSError("original partial write error")

    def fail_staging_unlink(path, *args, **kwargs):
        if path.suffix == ".part":
            raise PermissionError("temporary unlink error")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(os, "write", fail_write)
        failure.setattr(Path, "unlink", fail_staging_unlink)
        with pytest.raises(ArtifactError, match="original partial write error"):
            _managed_file(root, expires_at=200)
    assert len(list(root.glob(".save-*.part"))) == 1
    assert list(root.glob("*.txt")) == list(root.glob("*.json")) == []

    assert cleaner.cleanup_expired_generated_files(now_epoch=100, ttl_seconds=10) == _stats(
        staging_scanned=1, staging_deleted=1,
    )
    assert list(root.glob(".save-*.part")) == []


def test_cleanup_retries_orphan_unlink_error_and_leaves_unowned_names_alone(cleanup, monkeypatch):
    cleaner, root = cleanup
    orphan = root / f".save-{'a' * 32}.part"
    unrelated = root / ".save-not-a-staging-id.part"
    directory = root / f".save-{'b' * 32}.part"
    orphan.write_bytes(b"orphan")
    unrelated.write_bytes(b"keep")
    directory.mkdir()
    real_unlink = Path.unlink

    def fail_orphan(path, *args, **kwargs):
        if path == orphan:
            raise PermissionError("temporarily unavailable")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as failure:
        failure.setattr(Path, "unlink", fail_orphan)
        assert cleaner.cleanup_expired_generated_files(now_epoch=100, ttl_seconds=10) == _stats(
            staging_scanned=1, errors=1,
        )
    assert orphan.exists()
    assert cleaner.cleanup_expired_generated_files(now_epoch=100, ttl_seconds=10) == _stats(
        staging_scanned=1, staging_deleted=1,
    )
    assert not orphan.exists()
    assert unrelated.read_bytes() == b"keep"
    assert directory.is_dir()


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


@pytest.mark.parametrize("phase", ["partial", "manifest", "payload"])
def test_process_termination_releases_ownership_and_cleanup_preserves_retry(cleanup, phase):
    cleaner, root = cleanup
    process = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-c", _INTERRUPTED_WRITER, str(root), phase],
        cwd=Path(__file__).resolve().parents[2], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        announced = pool.submit(process.stdout.readline)
        try:
            assert announced.result(timeout=30).strip() == b"staged"
            assert cleaner.cleanup_expired_generated_files(now_epoch=101, ttl_seconds=10) == _stats(busy=1)
            assert len(list(root.glob(".save-*.part"))) == (1 if phase == "partial" else 2)
            manifests_before_cleanup = {path: path.read_bytes() for path in root.glob("*.json")}
            process.stdin.write(b"exit")
            process.stdin.flush()
            assert process.wait(timeout=20) == 73
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=20)
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()
    # Windows can signal process exit before releasing its file locks. Wait for
    # actual ownership, not elapsed time; the extra process bounds a broken lock
    # implementation without retrying cleanup or hiding a collector failure.
    subprocess.run(
        [sys._base_executable, "-c",
         "from pathlib import Path\nimport sys\n"
         "from src.infra.artifact_store_lock import artifact_store_lock\n"
         "with artifact_store_lock(Path(sys.argv[1])):\n    pass\n", str(root)],
        cwd=Path(__file__).resolve().parents[2], check=True, timeout=20, capture_output=True,
    )
    assert cleaner.cleanup_expired_generated_files(now_epoch=101, ttl_seconds=10) == _stats(
        scanned=int(phase == "payload"),
        staging_scanned=1 if phase == "partial" else 2, staging_deleted=1 if phase == "partial" else 2,
    )
    assert list(root.glob(".save-*.part")) == []
    assert {path: path.read_bytes() for path in root.glob("*.json")} == manifests_before_cleanup
    assert len(list(root.glob("*.txt"))) == int(phase == "payload")
    text = "interrupted answer"
    operation = SaveOperation.for_text(
        text, operation_id="killed-save", session_id="killed-session", request_id="request",
        contract_revision=1, target_kind="compose", source_hash="a" * 64, answer_hash="b" * 64,
    )
    with patch("src.infra.saved_artifacts.time", return_value=101.0):
        manifest, payload = save_artifact(root, text, operation, ttl_seconds=60)
    assert payload.read_bytes() == text.encode("utf-8-sig")
    assert manifest.artifact.expires_at == (161 if phase == "partial" else 160)
    assert len(list(root.glob("*.txt"))) == len(list(root.glob("*.json"))) == 1
    for path, original in manifests_before_cleanup.items():
        assert path.read_bytes() == original
