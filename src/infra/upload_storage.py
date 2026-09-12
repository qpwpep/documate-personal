"""Release server-owned upload versions without touching borrowed upload inputs."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.core.uploads import validate_session_id
from src.infra.runtime_paths import get_upload_session_dir, get_uploads_dir


logger = logging.getLogger(__name__)
_IDENTITY = re.compile(r"[0-9a-f]{32}")


def _unredirected(path: Path) -> bool:
    return not path.is_symlink() and not path.is_junction() and path.resolve() == path


@dataclass(frozen=True)
class UploadStorage:
    """A session's validated storage boundary, retained across context resets."""

    session_id: str
    root: Path

    @classmethod
    def bind(cls, session_id: str) -> UploadStorage:
        session_id = validate_session_id(session_id)
        uploads = get_uploads_dir().resolve()
        root = uploads / session_id
        if get_upload_session_dir(session_id).resolve() != root or not _unredirected(root):
            raise ValueError("Upload storage is outside its session boundary")
        return cls(session_id=session_id, root=root)

    def managed_base(self) -> Path:
        base = self.root / "objects"
        if not _unredirected(self.root) or not _unredirected(base):
            raise ValueError("Managed upload storage was redirected")
        return base

    def owns(self, path: Path) -> bool:
        base = self.managed_base()
        try:
            identity, version, _name = path.relative_to(base).parts
        except ValueError:
            return False
        return bool(
            _IDENTITY.fullmatch(identity) and _IDENTITY.fullmatch(version)
            and all(_unredirected(item) for item in (base / identity, path.parent, path))
            and (path.name == ".upload-part" or path.suffix.casefold() in {".py", ".ipynb"})
        )


def remove_managed_upload_files(storage: UploadStorage, paths: Iterable[str | Path]) -> None:
    """Best effort release; unreleased versions remain discoverable by reconciliation."""
    for value in paths:
        path = Path(value)
        try:
            if not storage.owns(path):
                continue
            path.unlink(missing_ok=True)
            # Only empty version/identity directories belong to this release.
            for folder in (path.parent, path.parent.parent):
                if _unredirected(folder):
                    try:
                        folder.rmdir()
                    except OSError:
                        pass
        except (OSError, RuntimeError, ValueError):
            logger.warning("upload_file_cleanup_failed", exc_info=True)


def reconcile_managed_upload_files(
    storage: UploadStorage, *, retained_paths: Iterable[str], protected_paths: Iterable[str] = (),
) -> None:
    """Reclaim unowned versions while the caller holds the existing session lock.

    Candidate creation uses that same lock. Borrowed inputs may be anywhere in
    the session directory, so a current request protects its entire input version.
    Staging and unknown directory layouts remain outside this ownership contract.
    """
    try:
        base = storage.managed_base()
        if not base.is_dir():
            return
        protected = {Path(value).resolve() for value in (*retained_paths, *protected_paths)}
        for identity in base.iterdir():
            if not _IDENTITY.fullmatch(identity.name) or not _unredirected(identity) or not identity.is_dir():
                continue
            for version in identity.iterdir():
                if (not _IDENTITY.fullmatch(version.name) or not _unredirected(version)
                        or not version.is_dir() or any(path.is_relative_to(version) for path in protected)):
                    continue
                children = list(version.iterdir())
                if any(not storage.owns(path) or not path.is_file() for path in children):
                    continue
                remove_managed_upload_files(storage, children)
                # Interrupted writes can also leave an empty version directory.
                if _unredirected(version):
                    try:
                        version.rmdir()
                    except OSError:
                        pass
            if _unredirected(identity):
                try:
                    identity.rmdir()
                except OSError:
                    pass
    except (OSError, RuntimeError, ValueError):
        logger.warning("upload_object_reconciliation_failed", exc_info=True)
