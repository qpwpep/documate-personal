from __future__ import annotations

from typing import Any
from pathlib import Path

from src.core.answer_schema import ActionReceipt
from src.core.slack_contract import SlackDelivery
from src.core.save_contract import SaveArtifact, SaveOperation


def _status(result: Any) -> str:
    return str(result.get("status") or "").strip().lower() if isinstance(result, dict) else ""


def _failure(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("error") or result.get("message") or "실행 결과를 확인할 수 없습니다.").strip()
    return "실행 결과를 확인할 수 없습니다."


def build_save_receipt(save_result: Any, *, operation: SaveOperation | None = None) -> ActionReceipt:
    status = _status(save_result)
    path = str(save_result.get("file_path") or "").strip() if isinstance(save_result, dict) else ""
    if status in {"success", "ok"} and path:
        # An adapter must not turn a tool's self-reported success into proof.
        from src.infra.saved_artifacts import ArtifactError, read_saved_artifact
        import src.infra.tools.save_text as save_tool_module

        try:
            reported_operation = SaveOperation.model_validate(save_result.get("operation"))
            artifact = SaveArtifact.model_validate(save_result.get("artifact"))
            if operation is not None and reported_operation != operation:
                raise ArtifactError("artifact_mismatch", "저장 결과가 요청한 작업과 다릅니다.")
            operation = operation or reported_operation
            root = save_tool_module.get_save_text_output_dir()
            candidate = Path(path)
            if candidate.parent.resolve() != root.resolve() or candidate.name != artifact.filename:
                raise ArtifactError("artifact_mismatch", "저장 결과의 파일 위치가 일치하지 않습니다.")
            manifest, payload = read_saved_artifact(root, artifact.filename)
            if manifest.operation != operation or manifest.artifact != artifact or not operation.matches_bytes(payload):
                raise ArtifactError("artifact_mismatch", "저장된 산출물이 요청한 본문과 다릅니다.")
            return ActionReceipt(kind="save_text", status="success", file_path=path,
                                 operation=operation, artifact=artifact, verification="verified")
        except ArtifactError as exc:
            unknown = exc.code == "artifact_unverifiable"
            return ActionReceipt(kind="save_text", status="unknown" if unknown else "error",
                                 operation=operation, verification="unverifiable" if unknown else "failed",
                                 error_code=exc.code, error=str(exc))
        except (TypeError, ValueError, OSError) as exc:
            return ActionReceipt(kind="save_text", status="unknown", operation=operation,
                                 verification="unverifiable", error_code="artifact_unverifiable",
                                 error=f"저장 결과를 검증할 수 없습니다: {exc}")
    if status == "skipped":
        return ActionReceipt(kind="save_text", status="skipped", operation=operation,
                             message=str(save_result.get("reason") or "저장을 보류했습니다."))
    unknown = status != "error"
    return ActionReceipt(kind="save_text", status="unknown" if unknown else "error", operation=operation,
                         verification="unverifiable" if unknown else "failed",
                         error_code=(str(save_result.get("error_code") or "") or
                                     ("artifact_unverifiable" if unknown else "write_failed"))
                         if isinstance(save_result, dict) else "artifact_unverifiable",
                         error=_failure(save_result))


def build_slack_receipt(delivery: SlackDelivery) -> ActionReceipt:
    if delivery.status == "sent":
        return ActionReceipt(kind="slack_notify", status="success", slack=delivery)
    if delivery.failure is None:
        raise ValueError("cannot report an unattempted Slack delivery")
    status = ("unknown" if delivery.status == "unknown" else
              "skipped" if delivery.failure.stage in {"input", "selection"} else "error")
    return ActionReceipt(
        kind="slack_notify", status=status, slack=delivery,
        message=delivery.failure.message if status == "skipped" else None,
        error=delivery.failure.message if status != "skipped" else None,
        error_code=f"SLACK_{delivery.failure.code.upper()}",
    )
