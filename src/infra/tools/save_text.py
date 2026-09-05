from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from src.infra.runtime_paths import get_save_text_output_dir


def build_save_text_tool() -> Callable[..., dict[str, Any]]:
    def save_text_to_file(content: str, filename_prefix: str = "response") -> dict[str, Any]:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = get_save_text_output_dir()
            output_path.mkdir(parents=True, exist_ok=True)

            normalized_prefix = str(filename_prefix or "response").strip() or "response"
            filename = f"{normalized_prefix}_{ts}.txt"
            filepath = output_path / filename
            filepath.write_text(content, encoding="utf-8-sig")
            byte_count = len(str(content or "").encode("utf-8-sig"))

            return {
                "status": "success",
                "message": f"Saved output to {filename}",
                "file_path": str(filepath),
                "bytes": byte_count,
            }
        except Exception as exc:
            raise RuntimeError(f"Failed to save file: {exc}") from exc

    return save_text_to_file
