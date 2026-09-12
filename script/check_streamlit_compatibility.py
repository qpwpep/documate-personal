from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_TESTS = [
    "tests/web/test_streamlit_upload_flow.py",
    "tests/web/test_streamlit_upload_handler.py",
    "tests/web/test_streamlit_api_client.py",
    "tests/web/test_streamlit_state.py",
    "tests/web/test_streamlit_page.py",
]
TEST_RUNNER = """
import sys
from importlib.metadata import version

expected = sys.argv[1]
actual = version("streamlit")
if actual != expected:
    print(f"Expected Streamlit {expected}, but installed {actual}.", file=sys.stderr)
    raise SystemExit(1)

print(f"Testing minimum supported Streamlit {actual}.", flush=True)
import pytest

raise SystemExit(pytest.main(sys.argv[2:]))
"""


def _minimum_streamlit_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)
    requirements = [
        requirement
        for requirement in project["project"]["dependencies"]
        if re.match(r"\s*streamlit(?=[^A-Za-z0-9._-]|$)", requirement, re.IGNORECASE)
    ]
    if len(requirements) != 1:
        raise ValueError("Expected exactly one Streamlit dependency in pyproject.toml.")
    match = re.fullmatch(
        r"\s*streamlit\s*>=\s*(\d+\.\d+\.\d+)\s*", requirements[0], re.IGNORECASE
    )
    if match is None:
        raise ValueError("Streamlit dependency must use the form streamlit>=X.Y.Z.")
    return match.group(1)


def main() -> int:
    try:
        minimum_version = _minimum_streamlit_version()
        result = subprocess.run(
            [
                "uv",
                "run",
                "--locked",
                "--isolated",
                "--group",
                "dev",
                "--with",
                f"streamlit=={minimum_version}",
                "python",
                "-c",
                TEST_RUNNER,
                minimum_version,
                "-q",
                *UI_TESTS,
            ],
            cwd=ROOT,
            check=False,
        )
        return result.returncode
    except (OSError, KeyError, ValueError) as error:
        print(f"Streamlit compatibility check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
