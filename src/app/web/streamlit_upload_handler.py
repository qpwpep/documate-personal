from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class UploadSyncResult:
    file_name: str | None
    changed: bool
    removed: bool
    error_message: str | None = None


def sync_uploaded_file(
    uploaded_file: Any,
    session_path: Path,
    current_file_name: str | None,
) -> UploadSyncResult:
    if uploaded_file is None:
        if not current_file_name:
            return UploadSyncResult(file_name=None, changed=False, removed=False)

        old_path = session_path / current_file_name
        try:
            old_path.unlink()
        except FileNotFoundError:
            pass
        return UploadSyncResult(file_name=None, changed=True, removed=True)

    if uploaded_file.name == current_file_name:
        return UploadSyncResult(
            file_name=current_file_name,
            changed=False,
            removed=False,
        )

    file_path_on_disk = session_path / uploaded_file.name
    try:
        if current_file_name:
            old_path = session_path / current_file_name
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass

        with open(file_path_on_disk, "wb") as file_obj:
            file_obj.write(uploaded_file.getbuffer())

        return UploadSyncResult(
            file_name=uploaded_file.name,
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
            error_message=f"파일 업로드 실패 : {exc}",
        )
