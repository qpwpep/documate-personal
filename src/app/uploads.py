from __future__ import annotations

import hashlib
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.core.uploads import UploadAddition, UploadManifest, UploadSyncRequest, normalized_upload_name
from src.core.upload_formats import ALL_UPLOAD_SUFFIXES


@dataclass(frozen=True)
class StagedUpload:
    path: str
    name: str
    size_bytes: int
    content_hash: str
    conflicting_file_id: str | None = None


@dataclass
class UploadStageResult:
    files: list[StagedUpload] = field(default_factory=list)
    unchanged_names: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def review_upload_changes(
    manifest: UploadManifest,
    *,
    files: Sequence[StagedUpload] = (),
    remove: Sequence[str] = (),
) -> tuple[list[StagedUpload], list[str]]:
    """Review a new intent against confirmed state without modifying staged bytes.

    Keep even now-identical files so callers can release all staging paths after
    confirmation. Approval belongs to the caller and must be requested again.
    """
    existing = {normalized_upload_name(item.name): item for item in manifest.files}
    reviewed = []
    for item in files:
        current = existing.get(normalized_upload_name(item.name))
        conflict = current.file_id if current is not None and current.content_hash != item.content_hash else None
        reviewed.append(replace(item, conflicting_file_id=conflict))
    current_ids = {item.file_id for item in manifest.files}
    return reviewed, [file_id for file_id in remove if file_id in current_ids]


def build_upload_sync_request(
    manifest: UploadManifest,
    *,
    files: Sequence[StagedUpload] = (),
    replace_file_ids: Collection[str] = (),
    remove: Sequence[str] = (),
    clear: bool = False,
) -> UploadSyncRequest:
    """Prepare one approved intent; retain the returned request unchanged for retries.

    The request owns its lists and contains no references to a mutable UI draft.
    Rebuilding is a new intent with a new operation ID, never a retry.
    """
    existing = {normalized_upload_name(item.name): item for item in manifest.files if item.file_id not in remove}
    additions = []
    for item in files:
        current = existing.get(normalized_upload_name(item.name))
        if current is not None and current.content_hash == item.content_hash:
            continue
        if current is not None and current.file_id not in replace_file_ids:
            raise ValueError("같은 이름의 파일 교체를 먼저 확인해 주세요.")
        additions.append(UploadAddition(
            path=item.path, name=item.name, content_hash=item.content_hash,
            replace_file_id=current.file_id if current is not None else None,
        ))
    return UploadSyncRequest(
        epoch=manifest.epoch, expected_revision=manifest.revision,
        operation_id=str(uuid4()), add=additions, remove=list(remove), clear=clear,
    )


def stage_uploaded_files(
    uploaded_files: list[Any],
    session_path: Path,
    *,
    existing_files: list[Any],
    max_files: int,
    max_file_mib: int,
    max_total_mib: int,
) -> UploadStageResult:
    """Stage a complete batch without changing any confirmed upload or source revision."""
    result = UploadStageResult()
    existing = {normalized_upload_name(item.name): item for item in existing_files}
    candidates: dict[str, tuple[str, bytes, str, Any]] = {}
    candidate_bytes = 0
    seen_hashes: dict[str, str] = {}
    for uploaded in uploaded_files:
        name = Path(str(uploaded.name).replace("\\", "/")).name
        if name in {"", ".", ".."} or Path(name).suffix.lower() not in ALL_UPLOAD_SUFFIXES:
            result.errors.append(f"{name or '(이름 없음)'}: .py · .ipynb · PDF · DOCX 또는 지원하는 이미지 파일을 첨부해 주세요.")
            continue
        try:
            advertised_size = getattr(uploaded, "size", None)
            if advertised_size is not None and advertised_size > max_file_mib * 1024 * 1024:
                raise ValueError(f"파일당 {max_file_mib} MiB 한도를 초과했습니다.")
            content = bytes(uploaded.getbuffer())
            if len(content) > max_file_mib * 1024 * 1024:
                raise ValueError(f"파일당 {max_file_mib} MiB 한도를 초과했습니다.")
        except Exception as exc:
            result.errors.append(f"{name}: 파일 준비 실패 ({exc})")
            continue
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        key = normalized_upload_name(name)
        previous = existing.get(key)
        if key in seen_hashes:
            if seen_hashes[key] != digest:
                result.errors.append(f"{name}: 한 번에 같은 이름의 서로 다른 파일을 추가할 수 없습니다.")
            else:
                result.unchanged_names.append(name)
            continue
        seen_hashes[key] = digest
        if previous is not None and previous.content_hash == digest:
            result.unchanged_names.append(name)
            continue
        candidate_bytes += len(content)
        if candidate_bytes > max_total_mib * 1024 * 1024:
            result.errors.append(f"전체 첨부 크기는 {max_total_mib} MiB 이하여야 합니다.")
            return result
        candidates[key] = (name, content, digest, previous)

    next_count = len(existing_files) + sum(previous is None for _, _, _, previous in candidates.values())
    next_size = sum(item.size_bytes for item in existing_files)
    next_size += sum(len(content) - (previous.size_bytes if previous is not None else 0) for _, content, _, previous in candidates.values())
    if next_count > max_files:
        result.errors.append(f"한 세션에는 최대 {max_files}개 파일을 첨부할 수 있습니다.")
    if next_size > max_total_mib * 1024 * 1024:
        result.errors.append(f"전체 첨부 크기는 {max_total_mib} MiB 이하여야 합니다.")
    if result.errors:
        return result

    try:
        for name, content, digest, previous in candidates.values():
            target_dir = session_path / "staging" / uuid4().hex
            target_dir.mkdir(parents=True, exist_ok=False)
            path = target_dir / name
            pending_path = target_dir / ".upload-part"
            try:
                pending_path.write_bytes(content)
                pending_path.replace(path)
            except Exception:
                pending_path.unlink(missing_ok=True)
                target_dir.rmdir()
                raise
            result.files.append(StagedUpload(
                path=str(path.resolve()), name=name, size_bytes=len(content), content_hash=digest,
                conflicting_file_id=previous.file_id if previous is not None else None,
            ))
    except Exception as exc:
        discard_staged_files(result.files, session_path)
        result.files = []
        result.errors.append(f"파일 저장 실패: {exc}")
    return result


def discard_staged_files(files: list[StagedUpload], session_path: Path) -> None:
    staging_root = (session_path / "staging").resolve()
    for item in files:
        path = Path(item.path).resolve()
        if not path.is_relative_to(staging_root):
            continue
        try:
            path.unlink(missing_ok=True)
            path.parent.rmdir()
        except OSError:
            # Abandoned staging files are also covered by the session cleanup policy.
            pass
