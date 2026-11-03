from __future__ import annotations

from src.core.answer_schema import ActionReceipt
from src.core.slack_contract import SlackDelivery
from src.core.save_contract import SaveOperation, SaveToolResult
from src.infra.saved_artifacts import ArtifactError, read_saved_artifact
from src.infra.tools import save_text


def build_save_receipt(
    save_result: SaveToolResult, *, operation: SaveOperation, invocation_id: str | None = None,
) -> ActionReceipt:
    # The typed result identifies the write; only reading storage proves its outcome.
    try:
        if save_result.operation != operation:
            raise ArtifactError("artifact_mismatch", "저장 결과가 요청한 작업과 다릅니다.")
        root = save_text.get_save_text_output_dir()
        if (save_result.file_path.parent.resolve() != root.resolve()
                or save_result.file_path.name != save_result.artifact.filename):
            raise ArtifactError("artifact_mismatch", "저장 결과의 파일 위치가 일치하지 않습니다.")
        manifest, payload = read_saved_artifact(root, save_result.artifact.filename)
        if (manifest.operation != operation or manifest.artifact != save_result.artifact
                or not operation.matches_bytes(payload)):
            raise ArtifactError("artifact_mismatch", "저장된 산출물이 요청한 본문과 다릅니다.")
        return ActionReceipt(kind="save_text", status="success", file_path=str(save_result.file_path),
                             operation=operation, artifact=save_result.artifact, verification="verified",
                             invocation_id=invocation_id)
    except ArtifactError as exc:
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


def build_slack_receipt(delivery: SlackDelivery) -> ActionReceipt:
    if delivery.status == "sent":
        return ActionReceipt(kind="slack_notify", status="success", slack=delivery, invocation_id=delivery.invocation_id)
    if delivery.failure is None:
        raise ValueError("cannot report an unattempted Slack delivery")
    status = ("unknown" if delivery.status == "unknown" else
              "skipped" if delivery.failure.stage in {"input", "selection"} else "error")
    return ActionReceipt(
        kind="slack_notify", status=status, slack=delivery, invocation_id=delivery.invocation_id,
        message=delivery.failure.message if status == "skipped" else None,
        error=delivery.failure.message if status != "skipped" else None,
        error_code=f"SLACK_{delivery.failure.code.upper()}",
    )
