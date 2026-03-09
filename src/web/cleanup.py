from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path, PureWindowsPath
from threading import Lock

from fastapi import HTTPException

from ..logging_utils import log_event
from ..runtime_paths import get_save_text_output_dir
from ..settings import AppSettings
from .session_store import InMemorySessionStore


logger = logging.getLogger(__name__)
ALLOWED_UPLOAD_SUFFIXES = {".py", ".ipynb"}


def resolve_download_path(output_dir: Path, filename: str) -> Path:
    base_dir = output_dir.resolve()
    normalized_filename = filename.replace("\\", "/")

    if Path(normalized_filename).is_absolute() or PureWindowsPath(normalized_filename).is_absolute():
        raise HTTPException(status_code=403, detail="Forbidden: Invalid file path")

    candidate = (base_dir / normalized_filename).resolve()
    try:
        candidate.relative_to(base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid file path") from exc
    return candidate


def validate_upload_file_path(upload_file_path: str | None, session_id: str) -> str | None:
    if not upload_file_path:
        return None

    try:
        session_upload_dir = (Path("uploads") / session_id).resolve()
        candidate_path = Path(upload_file_path).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid upload_file_path") from exc

    if session_upload_dir not in candidate_path.parents:
        raise HTTPException(status_code=400, detail="Invalid upload file location")

    if candidate_path.suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported upload file type")

    if not candidate_path.is_file():
        raise HTTPException(status_code=400, detail="Upload file not found")

    return str(candidate_path)


class RuntimeCleaner:
    def __init__(self, settings: AppSettings, session_store: InMemorySessionStore) -> None:
        self.settings = settings
        self.session_store = session_store
        self._lock = Lock()
        self._last_file_cleanup_monotonic = 0.0

    def collect_protected_session_ids(self, current_session_id: str | None) -> set[str]:
        protected_session_ids = self.session_store.active_session_ids()
        if current_session_id:
            protected_session_ids.add(current_session_id)
        return protected_session_ids

    @staticmethod
    def get_latest_mtime_epoch(path: Path) -> float:
        latest_mtime = 0.0
        try:
            latest_mtime = path.stat().st_mtime
        except OSError:
            return latest_mtime

        if not path.is_dir():
            return latest_mtime

        try:
            for child in path.rglob("*"):
                try:
                    child_mtime = child.stat().st_mtime
                    if child_mtime > latest_mtime:
                        latest_mtime = child_mtime
                except OSError:
                    continue
        except OSError:
            return latest_mtime

        return latest_mtime

    def cleanup_expired_upload_dirs(
        self,
        *,
        now_epoch: float,
        ttl_seconds: int,
        protected_session_ids: set[str],
    ) -> dict[str, int]:
        uploads_root = Path("uploads")
        stats = {
            "scanned": 0,
            "deleted": 0,
            "skipped_active_dirs": 0,
            "errors": 0,
        }

        if not uploads_root.exists():
            return stats

        try:
            session_entries = list(uploads_root.iterdir())
        except OSError as exc:
            log_event(logger, logging.WARNING, "upload_cleanup_scan_error", root=uploads_root, error=exc)
            stats["errors"] += 1
            return stats

        for session_dir in session_entries:
            if not session_dir.is_dir():
                continue

            stats["scanned"] += 1
            session_id = session_dir.name
            if session_id in protected_session_ids:
                stats["skipped_active_dirs"] += 1
                continue

            latest_mtime = self.get_latest_mtime_epoch(session_dir)
            if latest_mtime <= 0 or (now_epoch - latest_mtime) <= ttl_seconds:
                continue

            try:
                shutil.rmtree(session_dir)
                stats["deleted"] += 1
            except Exception as exc:
                log_event(logger, logging.WARNING, "upload_cleanup_error", path=session_dir, error=exc)
                stats["errors"] += 1

        return stats

    def cleanup_expired_generated_files(self, *, now_epoch: float, ttl_seconds: int) -> dict[str, int]:
        output_dir = get_save_text_output_dir()
        stats = {
            "scanned": 0,
            "deleted": 0,
            "errors": 0,
        }

        if not output_dir.exists():
            return stats

        try:
            txt_files = list(output_dir.glob("*.txt"))
        except OSError as exc:
            log_event(logger, logging.WARNING, "generated_file_scan_error", root=output_dir, error=exc)
            stats["errors"] += 1
            return stats

        for txt_file in txt_files:
            if not txt_file.is_file():
                continue

            stats["scanned"] += 1
            try:
                file_mtime = txt_file.stat().st_mtime
            except OSError as exc:
                log_event(logger, logging.WARNING, "generated_file_stat_error", path=txt_file, error=exc)
                stats["errors"] += 1
                continue

            if (now_epoch - file_mtime) <= ttl_seconds:
                continue

            try:
                txt_file.unlink()
                stats["deleted"] += 1
            except Exception as exc:
                log_event(logger, logging.WARNING, "generated_file_cleanup_error", path=txt_file, error=exc)
                stats["errors"] += 1

        return stats

    def run_once(self, *, force: bool, current_session_id: str | None = None) -> dict[str, int | bool]:
        now_monotonic = time.monotonic()
        result: dict[str, int | bool] = {
            "interval_skipped": False,
            "upload_dirs_scanned": 0,
            "upload_dirs_deleted": 0,
            "skipped_active_dirs": 0,
            "generated_files_scanned": 0,
            "generated_files_deleted": 0,
            "errors": 0,
        }

        try:
            with self._lock:
                interval_skipped = False
                if (
                    not force
                    and (now_monotonic - self._last_file_cleanup_monotonic)
                    < self.settings.file_cleanup_interval_seconds
                ):
                    result["interval_skipped"] = True
                    interval_skipped = True
                else:
                    now_epoch = time.time()
                    protected_session_ids = self.collect_protected_session_ids(current_session_id)
                    upload_stats = self.cleanup_expired_upload_dirs(
                        now_epoch=now_epoch,
                        ttl_seconds=self.settings.session_ttl_seconds,
                        protected_session_ids=protected_session_ids,
                    )
                    generated_stats = self.cleanup_expired_generated_files(
                        now_epoch=now_epoch,
                        ttl_seconds=self.settings.generated_file_ttl_seconds,
                    )
                    self._last_file_cleanup_monotonic = now_monotonic

                    errors = upload_stats["errors"] + generated_stats["errors"]
                    result = {
                        "interval_skipped": interval_skipped,
                        "upload_dirs_scanned": upload_stats["scanned"],
                        "upload_dirs_deleted": upload_stats["deleted"],
                        "skipped_active_dirs": upload_stats["skipped_active_dirs"],
                        "generated_files_scanned": generated_stats["scanned"],
                        "generated_files_deleted": generated_stats["deleted"],
                        "errors": errors,
                    }
        except Exception as exc:
            log_event(logger, logging.WARNING, "file_cleanup_unhandled_error", force=force, error=exc)
            result["errors"] = int(result.get("errors", 0)) + 1

        log_event(
            logger,
            logging.INFO,
            "file_cleanup_event",
            force=force,
            interval_skipped=result["interval_skipped"],
            upload_dirs_scanned=result["upload_dirs_scanned"],
            upload_dirs_deleted=result["upload_dirs_deleted"],
            skipped_active_dirs=result["skipped_active_dirs"],
            generated_files_scanned=result["generated_files_scanned"],
            generated_files_deleted=result["generated_files_deleted"],
            errors=result["errors"],
        )
        return result
