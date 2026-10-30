from __future__ import annotations

import argparse
import sys

from src.infra.runtime_encoding import maybe_reexec_with_utf8
from src.app.service_manager.web import run_web_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["startweb", "stopweb", "restartweb", "statusweb"],
        help="Start, stop, restart, or inspect the managed web services.",
    )
    parser.add_argument("--ui", choices=["react", "streamlit"],
                        help="UI to start (default: React; restart preserves the recorded UI).")
    return parser


def main(argv: list[str] | None = None) -> int:
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    maybe_reexec_with_utf8("src.app.service_manager", resolved_argv)
    parser = build_parser()
    args = parser.parse_args(resolved_argv)
    if args.ui is not None and args.mode in {"stopweb", "statusweb"}:
        parser.error("--ui is only supported with startweb or restartweb")
    return run_web_service(args.mode, ui=args.ui)
