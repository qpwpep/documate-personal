from __future__ import annotations

from collections.abc import Callable

from src.core.save_contract import SaveOperation, SaveToolResult
from src.infra.runtime_paths import get_save_text_output_dir
from src.infra.saved_artifacts import save_artifact


def build_save_text_tool(*, ttl_seconds: int = 86400) -> Callable[..., SaveToolResult]:
    def save_text_to_file(
        content: str, *, operation: SaveOperation,
    ) -> SaveToolResult:
        """Save exact UTF-8 BOM bytes under a storage-generated artifact name.

        The caller binds the save to its request and selected answer. Retrying
        the same operation with the same content returns the same artifact.
        """
        manifest, filepath = save_artifact(
            get_save_text_output_dir(), content, operation, ttl_seconds=ttl_seconds,
        )
        return SaveToolResult(file_path=filepath, operation=manifest.operation, artifact=manifest.artifact)

    return save_text_to_file
