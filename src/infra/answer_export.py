"""Choose a new export generation only after its previous retention has expired."""
from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from src.core.save_contract import SaveOperation
from src.infra import saved_artifacts
from src.infra.tools import save_text


def select_current_save_operation(operation: SaveOperation) -> SaveOperation:
    """Keep retries stable; an explicit new export can replace an expired file.

    Existing files and tombstones remain untouched. Unknown, damaged, or
    conflicting storage stays bound to its operation so saving fails closed.
    """
    root = save_text.get_save_text_output_dir()
    while True:
        try:
            manifest = saved_artifacts.read_saved_manifest(root, saved_artifacts.artifact_filename(operation))
        except (saved_artifacts.ArtifactError, OSError):
            return operation
        if manifest.operation != operation or saved_artifacts.time() < manifest.artifact.expires_at:
            return operation
        operation = SaveOperation.model_validate({
            **operation.model_dump(mode="json"),
            "operation_id": uuid5(NAMESPACE_URL, f"expired-answer-export:{operation.binding_sha256}").hex,
            "contract_revision": operation.contract_revision + 1,
        })
