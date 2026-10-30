"""Export browser contracts without starting services or contacting providers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.web.app import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Export FastAPI OpenAPI and SSE schemas for the browser client.")
    parser.add_argument("--check", action="store_true", help="Fail when the generated contract differs from the server")
    args = parser.parse_args()

    # Creating the app does not enter its lifespan or validate provider keys.
    expected = json.dumps(create_app().openapi(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    destination = ROOT / "frontend" / "src" / "api" / "openapi.json"
    if args.check:
        if not destination.exists() or destination.read_text(encoding="utf-8") != expected:
            print("Browser contract is outdated. Run: uv run --locked python script/export_web_contract.py")
            return 1
        print("Browser contract matches FastAPI.")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(expected, encoding="utf-8", newline="\n")
    print(f"Updated {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
