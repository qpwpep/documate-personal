from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infra.runtime_paths import get_env_example_path, get_runtime_reference_path
from src.infra.settings import DEFAULT_BENCHMARK_CONFIG_PATH
from src.infra.settings_sync import build_env_example_text, build_unified_diff, sync_runtime_reference_settings_sections


def _sync_env_example(check: bool) -> bool:
    path = get_env_example_path()
    expected = build_env_example_text(DEFAULT_BENCHMARK_CONFIG_PATH)
    actual = path.read_text(encoding="utf-8") if path.exists() else ""
    if check:
        if actual != expected:
            print(build_unified_diff(path, actual, expected))
            return False
        return True
    path.write_text(expected, encoding="utf-8", newline="\n")
    print(f"Updated {path}")
    return True


def _sync_runtime_reference(check: bool) -> bool:
    path = get_runtime_reference_path()
    actual = path.read_text(encoding="utf-8")
    expected = sync_runtime_reference_settings_sections(actual, DEFAULT_BENCHMARK_CONFIG_PATH)
    if check:
        if actual != expected:
            print(build_unified_diff(path, actual, expected))
            return False
        return True
    path.write_text(expected, encoding="utf-8", newline="\n")
    print(f"Updated {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync .env.example and runtime reference settings tables.")
    parser.add_argument("--check", action="store_true", help="Fail if generated content does not match tracked files")
    args = parser.parse_args()

    results = [
        _sync_env_example(check=args.check),
        _sync_runtime_reference(check=args.check),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
