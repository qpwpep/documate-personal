from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.core.save_contract import SaveOperation
from src.infra.runtime_paths import get_save_text_output_dir
from src.infra.saved_artifacts import save_artifact


def build_save_text_tool(*, ttl_seconds: int = 86400) -> Callable[..., dict[str, Any]]:
    def save_text_to_file(
        content: str, *, operation: SaveOperation,
    ) -> dict[str, Any]:
        """Save exact UTF-8 BOM bytes under a storage-generated artifact name.

        The caller binds the save to its request and selected answer. Retrying
        the same operation with the same content returns the same artifact.
        """
        manifest, filepath = save_artifact(
            get_save_text_output_dir(), content, operation, ttl_seconds=ttl_seconds,
        )
        return {
            "status": "success", "message": f"Saved output to {manifest.artifact.filename}",
            "file_path": str(filepath), "bytes": manifest.artifact.byte_count,
            "operation": manifest.operation.model_dump(mode="json"),
            "artifact": manifest.artifact.model_dump(mode="json"), "verification": "verified",
        }

    return save_text_to_file
