from __future__ import annotations

from collections.abc import Callable
import hashlib
from typing import Any
from uuid import uuid4

from src.core.save_contract import SaveOperation
from src.infra.runtime_paths import get_save_text_output_dir
from src.infra.saved_artifacts import save_artifact


def build_save_text_tool(*, ttl_seconds: int = 86400) -> Callable[..., dict[str, Any]]:
    def save_text_to_file(
        content: str, filename_prefix: str = "response", *, operation: SaveOperation | None = None,
    ) -> dict[str, Any]:
        """Save exact text; filename_prefix remains only for call compatibility."""
        if operation is None:
            identifier = uuid4().hex
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            operation = SaveOperation.for_text(
                content, operation_id=identifier, session_id="direct", request_id=identifier,
                contract_revision=1, target_kind="compose", source_hash=content_hash,
                answer_hash=content_hash,
            )
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
