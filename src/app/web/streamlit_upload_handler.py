from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.core.uploads import normalized_upload_name


@dataclass
class UploadSyncResult:
    file_name: str | None
    changed: bool
    removed: bool
    error_message: str | None = None


@dataclass
class StagedUpload:
    path: str
    name: str
    size_bytes: int
    content_hash: str
    conflicting_file_id: str | None = None
    replace_file_id: str | None = None


@dataclass
class UploadStageResult:
    files: list[StagedUpload] = field(default_factory=list)
    unchanged_names: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class PendingUploadOperation:
    epoch: str
    expected_revision: int
    operation_id: str = field(default_factory=lambda: str(uuid4()))
    files: list[StagedUpload] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    clear: bool = False
    prompt: str | None = None
    error: str | None = None
    needs_refresh_review: bool = False
    attempted: bool = False
    failed: bool = False

    def request_payload(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "expected_revision": self.expected_revision,
            "operation_id": self.operation_id,
            "add": [{"path": item.path, "name": item.name, "replace_file_id": item.replace_file_id} for item in self.files],
            "remove": list(self.remove),
            "clear": self.clear,
        }


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
        if name in {"", ".", ".."} or Path(name).suffix.lower() not in {".py", ".ipynb"}:
            result.errors.append(f"{name or '(이름 없음)'}: .py 또는 .ipynb 파일만 첨부할 수 있습니다.")
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


def sync_uploaded_file(
    uploaded_file: Any,
    session_path: Path,
    current_file_name: str | None,
) -> UploadSyncResult:
    safe_current_file_name = Path(str(current_file_name)).name if current_file_name else None

    if uploaded_file is None:
        if not safe_current_file_name or safe_current_file_name in {".", ".."}:
            return UploadSyncResult(file_name=None, changed=False, removed=False)

        old_path = session_path / safe_current_file_name
        try:
            old_path.unlink()
        except FileNotFoundError:
            pass
        return UploadSyncResult(file_name=None, changed=True, removed=True)

    safe_file_name = Path(str(uploaded_file.name)).name
    if safe_file_name in {"", ".", ".."}:
        return UploadSyncResult(
            file_name=None,
            changed=False,
            removed=False,
            error_message="Invalid upload filename",
        )

    file_path_on_disk = session_path / safe_file_name
    try:
        content = bytes(uploaded_file.getbuffer())
        if safe_file_name == safe_current_file_name and file_path_on_disk.is_file():
            if file_path_on_disk.read_bytes() == content:
                return UploadSyncResult(file_name=safe_file_name, changed=False, removed=False)

        file_path_on_disk.write_bytes(content)

        if safe_current_file_name and safe_current_file_name not in {safe_file_name, ".", ".."}:
            old_path = session_path / safe_current_file_name
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass

        return UploadSyncResult(
            file_name=safe_file_name,
            changed=True,
            removed=False,
        )
    except ValueError as exc:
        return UploadSyncResult(
            file_name=None,
            changed=False,
            removed=False,
            error_message=f"파일 업로드 실패 (내용 오류): {exc}",
        )
    except Exception as exc:
        return UploadSyncResult(
            file_name=None,
            changed=False,
            removed=False,
            error_message=f"파일 업로드 실패: {exc}",
        )
