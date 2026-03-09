from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping


_REEXEC_GUARD_ENV = "_DOCUMATE_UTF8_REEXECED"


def ensure_utf8_stdio() -> None:
    """Best-effort UTF-8 reconfiguration for standard IO streams."""
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue

        try:
            kwargs = {"encoding": "utf-8"}
            if stream_name != "stdin":
                kwargs["errors"] = "replace"
            reconfigure(**kwargs)
        except Exception:
            # Keep runtime stable even when stream does not support runtime reconfiguration.
            continue


def build_utf8_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return environment variables forcing UTF-8 mode for child Python processes."""
    env = dict(base_env) if base_env is not None else dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def maybe_reexec_with_utf8(module_name: str, argv: list[str]) -> None:
    """Relaunch `python -m <module_name>` with UTF-8 mode when needed."""
    if sys.flags.utf8_mode == 1:
        return

    if os.environ.get(_REEXEC_GUARD_ENV) == "1":
        return

    env = build_utf8_env(os.environ)
    env[_REEXEC_GUARD_ENV] = "1"
    cmd = [sys.executable, "-X", "utf8", "-m", module_name, *argv]
    raise SystemExit(subprocess.call(cmd, env=env))
