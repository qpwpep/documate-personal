from __future__ import annotations

from src.infra.rag_build import run_cli as _run_cli


def run_cli() -> int:
    return _run_cli()


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
