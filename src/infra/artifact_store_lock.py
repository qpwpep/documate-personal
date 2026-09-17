"""OS-owned exclusion for local artifact writers and their garbage collector."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import stat
from time import sleep

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def _try_lock(descriptor: int) -> bool:
    try:
        if os.name == "nt":
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return False
        raise
    return True


@contextmanager
def artifact_store_lock(output_dir: Path, *, blocking: bool = True) -> Iterator[bool]:
    """Use a distinct OS handle per caller to exclude both threads and processes.

    The permanent lock file must never be unlinked: replacing its inode could
    split owners between two independent locks. The OS releases ownership when
    a process exits, including termination without Python cleanup. Unsupported
    locking fails closed instead of falling back to stale lock-file ownership.
    """
    root = output_dir.resolve()
    lock_path = root / ".save.lock"
    if lock_path.is_symlink() or lock_path.resolve().parent != root:
        raise OSError("Artifact store lock must be a regular file inside its storage directory")
    descriptor = os.open(
        lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    acquired = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("Artifact store lock must be a regular file")
        while not (acquired := _try_lock(descriptor)):
            if not blocking:
                yield False
                return
            # Only writers wait. Cleanup skips busy stores, regardless of age.
            sleep(0.05)
        yield True
    finally:
        try:
            if acquired:
                if os.name == "nt":
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
