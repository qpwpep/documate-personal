"""Run the repository's offline regression gates from one entry point."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-browser", action="store_true", help="Run all checks except Chromium acceptance tests")
    args = parser.parse_args()
    npm = shutil.which("npm")
    uv = shutil.which("uv")
    if npm is None or uv is None:
        parser.error("uv and npm must be installed before running verification")
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "LIVE_TEST": "false",
        "RUN_DOCLING_TESTS": "0",
        "RUN_LIVE_API_AVAILABILITY": "0",
        "RUN_LIVE_CITATION_SUPPORT": "0",
        "RUN_LIVE_CODE_BINDING_SUFFICIENCY": "0",
        "RUN_LIVE_EXACT_INPUT_VALIDATION": "0",
        "RUN_LIVE_EXTRACT_LOCATION_FULFILLMENT": "0",
        "RUN_LIVE_INFERENCE_API_IDENTITY": "0",
        "RUN_LIVE_PARTIAL_ANSWER_SCOPE": "0",
        "RUN_LIVE_PLANNER_COPY_OBLIGATIONS": "0",
        "RUN_LIVE_PLANNER_SOURCE_LOCATIONS": "0",
        "RUN_LIVE_PLANNER_SYMBOL_IDENTITY": "0",
        "RUN_LIVE_POST_SOURCE_QUOTE": "0",
        "RUN_LIVE_TRANSFORM_ANSWER_DELIVERY": "0",
    }
    commands = [
        [uv, "lock", "--check"],
        [sys.executable, "script/check_encoding.py"],
        [sys.executable, "script/sync_env_example.py", "--check"],
        [sys.executable, "script/export_web_contract.py", "--check"],
        [npm, "--prefix", "frontend", "run", "check:api"],
        [npm, "--prefix", "frontend", "run", "typecheck"],
        [npm, "--prefix", "frontend", "run", "lint"],
        [sys.executable, "-m", "pytest", "-q"],
        [npm, "--prefix", "frontend", "run", "test:unit"],
    ]
    if not args.skip_browser:
        commands.append([npm, "--prefix", "frontend", "run", "test:e2e"])
    for command in commands:
        print(f"\nRunning: {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
