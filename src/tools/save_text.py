from __future__ import annotations

from datetime import datetime

from langchain_core.tools import StructuredTool

from ..runtime_paths import get_save_text_output_dir
from ._common import SaveArgs


def build_save_text_tool():
    def save_text_to_file(content: str, filename_prefix: str = "response") -> dict:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = get_save_text_output_dir()
            output_path.mkdir(parents=True, exist_ok=True)

            normalized_prefix = str(filename_prefix or "response").strip() or "response"
            filename = f"{normalized_prefix}_{ts}.txt"
            filepath = output_path / filename
            filepath.write_text(content, encoding="utf-8")

            return {
                "message": f"Saved output to {filename}",
                "file_path": str(filepath),
            }
        except Exception as exc:
            raise RuntimeError(f"Failed to save file: {exc}") from exc

    return StructuredTool.from_function(
        name="save_text",
        description=(
            "Save the given final response text into a timestamped .txt file in the current directory. "
            "Call this at most ONCE per user request. If you already saved, do not call again."
        ),
        func=save_text_to_file,
        args_schema=SaveArgs,
    )
