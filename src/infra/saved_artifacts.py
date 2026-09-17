"""Immutable local save operations with verified bytes and no-replace publication.

A manifest reserves an operation's payload. It is not a commit receipt: readers
must verify both the manifest and the complete payload before returning bytes.
Manifest tombstones outlive expired payloads so retries cannot extend retention.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
from time import time
from uuid import uuid4

from pydantic import ValidationError

from src.core.save_contract import SaveArtifact, SaveManifest, SaveOperation
from src.infra.artifact_store_lock import artifact_store_lock
from src.infra.logging_utils import log_event


logger = logging.getLogger(__name__)
_STAGING_NAME = re.compile(r"\.save-[0-9a-f]{32}\.part")


class ArtifactError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def manifest_path_for(artifact_path: Path) -> Path:
    return artifact_path.with_name(artifact_path.name + ".json")


def _operation_identifier(operation: SaveOperation) -> str:
    return hashlib.sha256(json.dumps(
        [operation.session_id, operation.operation_id], separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def artifact_filename(operation: SaveOperation) -> str:
    """Resolve the public name without requiring a successful save receipt."""
    return f"response_{_operation_identifier(operation)}.txt"


def _artifact_path(output_dir: Path, filename: str) -> Path:
    if (not filename or Path(filename).name != filename or not filename.endswith(".txt")
            or any(character in filename for character in (":", "/", "\\"))):
        raise ArtifactError("artifact_missing", "Invalid artifact filename")
    root = output_dir.resolve()
    path = root / filename
    manifest_path = manifest_path_for(path)
    if (path.is_symlink() or manifest_path.is_symlink()
            or path.resolve().parent != root or manifest_path.resolve().parent != root):
        raise ArtifactError("artifact_unverifiable", "Artifact path is outside its storage directory")
    return path


def _load_manifest(path: Path) -> SaveManifest:
    try:
        manifest = SaveManifest.model_validate_json(manifest_path_for(path).read_bytes())
    except FileNotFoundError as exc:
        raise ArtifactError("manifest_missing", "No saved operation manifest exists") from exc
    except (ValidationError, ValueError) as exc:
        raise ArtifactError("artifact_mismatch", "Saved operation manifest is invalid") from exc
    except OSError as exc:
        raise ArtifactError("artifact_unverifiable", f"Cannot read saved operation manifest: {exc}") from exc
    if (manifest.artifact.filename != path.name
            or manifest.artifact.artifact_id != _operation_identifier(manifest.operation)
            or path.name != artifact_filename(manifest.operation)
            or manifest.artifact.sha256 != manifest.operation.payload_sha256
            or manifest.artifact.byte_count != manifest.operation.byte_count):
        raise ArtifactError("artifact_mismatch", "Saved operation and artifact metadata disagree")
    return manifest


def _check_expiry(manifest: SaveManifest, now_epoch: float) -> None:
    if now_epoch >= manifest.artifact.expires_at:
        raise ArtifactError("artifact_expired", "This saved artifact has expired")


def read_saved_artifact(
    output_dir: Path, filename: str, *, now_epoch: float | None = None,
) -> tuple[SaveManifest, bytes]:
    """Read and verify the same bytes a caller may serve to its user."""
    path = _artifact_path(Path(output_dir), filename)
    manifest = _load_manifest(path)
    _check_expiry(manifest, time() if now_epoch is None else now_epoch)
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise ArtifactError("artifact_missing", "The saved artifact does not exist") from exc
    except OSError as exc:
        raise ArtifactError("artifact_unverifiable", f"Cannot read saved artifact: {exc}") from exc
    if len(payload) != manifest.artifact.byte_count or hashlib.sha256(payload).hexdigest() != manifest.artifact.sha256:
        raise ArtifactError("artifact_mismatch", "The saved artifact differs from its committed bytes")
    return manifest, payload


def _write_staging(root: Path, payload: bytes) -> Path:
    path = root / f".save-{uuid4().hex}.part"
    descriptor = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Writing the saved artifact made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if path.read_bytes() != payload:
            raise ArtifactError("artifact_mismatch", "Staged artifact failed its readback check")
        return path
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        _remove_staging(path)
        raise


def _remove_staging(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        # The next sweep takes the same store lock and reclaims this orphan.
        # Cleanup must not mask a write error or invalidate a verified save.
        log_event(logger, logging.WARNING, "artifact_staging_cleanup_error", path=path, error=exc)


def _publish(staging: Path, destination: Path) -> None:
    """Hard-link publication is atomic and cannot replace an existing name.

    Both paths belong to one local filesystem. Unsupported filesystems fail
    closed; falling back to replacing a destination would break preservation.
    """
    try:
        os.link(staging, destination)
    except FileExistsError:
        pass


def save_artifact(
    output_dir: Path, content: str, operation: SaveOperation, *,
    ttl_seconds: int,
) -> tuple[SaveManifest, Path]:
    payload = content.encode("utf-8-sig")
    if (hashlib.sha256(payload).hexdigest() != operation.payload_sha256
            or len(payload) != operation.byte_count):
        raise ArtifactError("idempotency_conflict", "Save operation does not identify these exact bytes")
    if ttl_seconds <= 0:
        raise ValueError("Artifact retention must be positive")
    root = Path(output_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
        with artifact_store_lock(root):
            return _save_artifact_locked(root, payload, operation, ttl_seconds=ttl_seconds)
    except OSError as exc:
        raise ArtifactError("write_failed", f"Failed to save file: {exc}") from exc


def _save_artifact_locked(
    root: Path, payload: bytes, operation: SaveOperation, *, ttl_seconds: int,
) -> tuple[SaveManifest, Path]:
    identifier = _operation_identifier(operation)
    filename = artifact_filename(operation)
    staged_payload = staged_manifest = None
    try:
        path = _artifact_path(root, filename)
        root = path.parent
        try:
            manifest = _load_manifest(path)
        except ArtifactError as exc:
            if exc.code != "manifest_missing":
                raise
            manifest = None
        if manifest is not None:
            if manifest.operation != operation:
                raise ArtifactError("idempotency_conflict", "This save operation already belongs to a different payload or request")
            _check_expiry(manifest, time())
            try:
                verified, _ = read_saved_artifact(root, filename)
                return verified, path
            except ArtifactError as exc:
                if exc.code != "artifact_missing":
                    raise

        staged_payload = _write_staging(root, payload)
        if manifest is None:
            created_at = time()
            manifest = SaveManifest(operation=operation, artifact=SaveArtifact(
                artifact_id=identifier, filename=filename, sha256=operation.payload_sha256,
                byte_count=operation.byte_count, created_at=created_at,
                expires_at=created_at + ttl_seconds,
            ))
            staged_manifest = _write_staging(root, manifest.model_dump_json().encode("utf-8"))
            _publish(staged_manifest, manifest_path_for(path))
            manifest = _load_manifest(path)
            if manifest.operation != operation:
                raise ArtifactError("idempotency_conflict", "A concurrent save reserved this operation for different bytes")
        _check_expiry(manifest, time())
        _publish(staged_payload, path)
        verified, _ = read_saved_artifact(root, filename)
        return verified, path
    finally:
        for staged in (staged_payload, staged_manifest):
            if staged is not None:
                _remove_staging(staged)


def cleanup_saved_artifacts(output_dir: Path, *, now_epoch: float, ttl_seconds: int) -> dict[str, int]:
    """Reclaim unowned staging and expired payloads without removing tombstones."""
    stats = {"scanned": 0, "deleted": 0, "errors": 0, "staging_scanned": 0, "staging_deleted": 0, "busy": 0}
    root = Path(output_dir).resolve()
    if not root.exists():
        return stats
    try:
        with artifact_store_lock(root, blocking=False) as acquired:
            if not acquired:
                stats["busy"] = 1
                return stats
            for path in root.iterdir():
                if _STAGING_NAME.fullmatch(path.name):
                    # Owning the store lock proves no writer can still use this
                    # private name. Its mtime and any public hard link are irrelevant.
                    if path.is_symlink() or not path.is_file():
                        continue
                    stats["staging_scanned"] += 1
                    try:
                        path.unlink(missing_ok=True)
                        stats["staging_deleted"] += 1
                    except OSError as exc:
                        stats["errors"] += 1
                        log_event(logger, logging.WARNING, "artifact_staging_cleanup_error", path=path, error=exc)
                elif path.match("*.txt") and path.is_file():
                    stats["scanned"] += 1
                    try:
                        sidecar = manifest_path_for(path)
                        if (path.is_symlink() or sidecar.is_symlink()
                                or path.resolve().parent != root or sidecar.resolve().parent != root):
                            raise ValueError("Generated artifact paths must stay inside their storage directory")
                        if sidecar.exists():
                            manifest = SaveManifest.model_validate_json(sidecar.read_bytes())
                            if manifest.artifact.filename != path.name:
                                raise ValueError("Saved artifact filename differs from its retention manifest")
                            expired = now_epoch >= manifest.artifact.expires_at
                        else:
                            # Preserve the existing mtime policy for legacy outputs.
                            expired = (now_epoch - path.stat().st_mtime) > ttl_seconds
                        if expired:
                            path.unlink()
                            stats["deleted"] += 1
                            # The immutable manifest stays: retries may not recreate
                            # expired payloads or extend their original retention.
                    except (OSError, ValueError) as exc:
                        stats["errors"] += 1
                        log_event(logger, logging.WARNING, "generated_file_cleanup_error", path=path, error=exc)
    except OSError as exc:
        stats["errors"] += 1
        log_event(logger, logging.WARNING, "generated_file_scan_error", root=root, error=exc)
    return stats
