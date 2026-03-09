from __future__ import annotations

import subprocess
import sys
from pathlib import Path


UTF8_BOM = b"\xef\xbb\xbf"
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".ipynb",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {
    ".editorconfig",
    ".env",
    ".env.example",
    ".gitattributes",
    ".gitignore",
    ".python-version",
}


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [
        Path(entry.decode("utf-8", errors="surrogateescape"))
        for entry in result.stdout.split(b"\x00")
        if entry
    ]


def _is_text_file(path: Path) -> bool:
    if path.name in TEXT_FILENAMES:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def main() -> int:
    checked_count = 0
    bom_files: list[str] = []
    non_utf8_files: list[str] = []

    for path in _tracked_files():
        if not path.is_file() or not _is_text_file(path):
            continue

        checked_count += 1
        raw = path.read_bytes()

        if raw.startswith(UTF8_BOM):
            bom_files.append(path.as_posix())

        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            non_utf8_files.append(path.as_posix())

    if non_utf8_files or bom_files:
        print("Encoding check failed.")
        if non_utf8_files:
            print("\n[Non UTF-8 files]")
            for file_path in non_utf8_files:
                print(f"- {file_path}")
        if bom_files:
            print("\n[UTF-8 BOM files]")
            for file_path in bom_files:
                print(f"- {file_path}")
        return 1

    print(f"Encoding check passed. Checked {checked_count} tracked text files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
