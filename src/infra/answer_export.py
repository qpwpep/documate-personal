"""Choose a new export generation only after its previous retention has expired."""
from __future__ import annotations

from collections.abc import Callable
from uuid import NAMESPACE_URL, uuid5

from src.core.answer_export import build_answer_export_operation
from src.core.answer_schema import ActionReceipt, AnswerResponse, export_answer_text
from src.core.save_contract import SaveOperation, SaveToolResult
from src.infra import saved_artifacts
from src.infra.tools import save_text


def select_answer_save_operation(
    answer: AnswerResponse, *, session_id: str,
    remembered: SaveOperation | None = None, pending: SaveOperation | None = None,
) -> SaveOperation:
    """HTTP exports and copied-answer requests share the same frozen save obligation."""
    operation = remembered or pending or build_answer_export_operation(answer, session_id=session_id)
    payload = export_answer_text(answer, include_sources=True).encode("utf-8-sig")
    if (operation.session_id != session_id or operation.answer_hash != answer.content_hash
            or not operation.matches_bytes(payload)):
        raise saved_artifacts.ArtifactError("idempotency_conflict", "선택한 저장 작업의 본문과 세션이 다릅니다.")
    return select_current_save_operation(operation)


def execute_save(
    save: Callable[[], SaveToolResult], *, operation: SaveOperation, invocation_id: str | None = None,
) -> ActionReceipt:
    """Translate expected storage failures and verify the bytes before reporting success."""
    try:
        result = save()
    except (saved_artifacts.ArtifactError, OSError) as exc:
        return build_save_failure(
            exc.code if isinstance(exc, saved_artifacts.ArtifactError) else "write_failed",
            str(exc), operation=operation, invocation_id=invocation_id,
        )
    return build_save_receipt(result, operation=operation, invocation_id=invocation_id)


def build_save_receipt(
    save_result: SaveToolResult, *, operation: SaveOperation, invocation_id: str | None = None,
) -> ActionReceipt:
    try:
        if save_result.operation != operation:
            raise saved_artifacts.ArtifactError("artifact_mismatch", "저장 결과가 요청한 작업과 다릅니다.")
        root = save_text.get_save_text_output_dir()
        if (save_result.file_path.parent.resolve() != root.resolve()
                or save_result.file_path.name != save_result.artifact.filename):
            raise saved_artifacts.ArtifactError("artifact_mismatch", "저장 결과의 파일 위치가 일치하지 않습니다.")
        manifest, _ = saved_artifacts.read_saved_artifact(root, save_result.artifact.filename)
        if manifest.operation != operation or manifest.artifact != save_result.artifact:
            raise saved_artifacts.ArtifactError("artifact_mismatch", "저장된 산출물이 요청한 본문과 다릅니다.")
        return ActionReceipt(kind="save_text", status="success", file_path=str(save_result.file_path),
                             operation=operation, artifact=save_result.artifact, verification="verified",
                             invocation_id=invocation_id)
    except saved_artifacts.ArtifactError as exc:
        return build_save_failure(exc.code, str(exc), operation=operation, invocation_id=invocation_id)
    except OSError as exc:
        return build_save_failure("artifact_unverifiable", f"저장 결과를 검증할 수 없습니다: {exc}",
                                  operation=operation, invocation_id=invocation_id)


def build_save_failure(
    error_code: str, message: str, *, operation: SaveOperation | None,
    invocation_id: str | None = None,
) -> ActionReceipt:
    unknown = error_code == "artifact_unverifiable"
    return ActionReceipt(kind="save_text", status="unknown" if unknown else "error", operation=operation,
                         verification="unverifiable" if unknown else "failed", error_code=error_code,
                         error=message, invocation_id=invocation_id)


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
