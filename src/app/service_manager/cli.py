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
        choices=["startweb", "stopweb"],
        help="Service mode: startweb or stopweb.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    maybe_reexec_with_utf8("src.app.service_manager", resolved_argv)
    parser = build_parser()
    args = parser.parse_args(resolved_argv)
    return run_web_service(args.mode)
