"""Export a selected server answer without synthesizing or replacing its body."""
from __future__ import annotations

import logging

from fastapi import HTTPException

from src.app.web.session_store import InMemorySessionStore, SessionUnavailable
from src.core.answer_export import answer_export_key, build_answer_export_operation
from src.core.answer_schema import AnswerResponse
from src.core.answer_schema.rendering import render_answer_text
from src.core.contracts.outcome import AnswerExportResult
from src.infra.logging_utils import log_event
from src.infra.saved_artifacts import ArtifactError
from src.infra.answer_export import select_current_save_operation
from src.infra.settings import AppSettings
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.nodes.actions.receipts import build_save_failure, build_save_receipt
from src.runtime.agent_runtime.session_context import SessionContext


logger = logging.getLogger(__name__)


class AnswerExportService:
    def __init__(self, *, settings: AppSettings, session_store: InMemorySessionStore):
        self._session_store = session_store
        self._save = build_save_text_tool(ttl_seconds=settings.generated_file_ttl_seconds)

    def save(self, session_id: str, answer_id: str, *, request_id: str) -> AnswerExportResult:
        try:
            with self._session_store.locked_existing_session(session_id) as entry:
                session = entry.agent.session
                snapshot = session.get_answer(answer_id)
                if snapshot is None:
                    raise HTTPException(status_code=404, detail={
                        "code": "answer_unavailable",
                        "message": "저장할 답변의 서버 기록이 만료되었거나 없습니다. 화면의 기존 답변은 계속 확인할 수 있습니다.",
                    })
                return self._save_snapshot(session_id, answer_id, snapshot, session=session, request_id=request_id)
        except SessionUnavailable as exc:
            raise HTTPException(status_code=410 if exc.expired else 404, detail={
                "code": "session_expired" if exc.expired else "session_unavailable",
                "message": "이 대화의 서버 기록이 만료되었거나 없습니다. 기존 화면 기록은 읽기 전용으로 확인해 주세요.",
            }) from exc

    def _save_snapshot(
        self, session_id: str, answer_id: str, snapshot: AnswerResponse, *,
        session: SessionContext, request_id: str,
    ) -> AnswerExportResult:
        body = render_answer_text(snapshot, include_sources=True)
        key = answer_export_key(snapshot)
        pending = session.pending_action
        matching_pending = bool(
            pending is not None and pending.body_prepared and pending.response is not None
            and pending.contract.action_requested("save_text")
            and answer_export_key(pending.response) == key
        )
        operation = session.answer_export_operations.get(key)
        prior = pending.save_operation if matching_pending else None
        if (operation is None and prior is not None and prior.session_id == session.session_id
                and prior.answer_hash == snapshot.content_hash and prior.matches_bytes(body.encode("utf-8-sig"))):
            operation = prior
        operation = select_current_save_operation(operation or build_answer_export_operation(snapshot, session_id=session.session_id))
        session.remember_answer_export(key, operation)
        try:
            result = self._save(body, operation=operation)
        except (ArtifactError, OSError) as exc:
            receipt = build_save_failure(
                exc.code if isinstance(exc, ArtifactError) else "write_failed",
                "파일을 만들지 못했습니다. 기존 답변은 유지됩니다.", operation=operation,
            )
            log_event(logger, logging.WARNING, "answer_export_error", session_id=session_id[:8],
                      request_id=request_id, answer_id=answer_id, error_code=receipt.error_code,
                      exception_type=type(exc).__name__)
        else:
            receipt = build_save_receipt(result, operation=operation)
        completed = receipt.status == "success"
        if completed and matching_pending:
            completed_actions = set(pending.completed_actions) | {"save_text"}
            waiting_slack = (pending.contract.actions.slack_notify.intent in {"requested", "unresolved"}
                             and "slack_notify" not in completed_actions)
            session.pending_action = pending.model_copy(update={
                "completed_actions": tuple(sorted(completed_actions)),
                "save_operation": operation, "save_receipt": receipt,
            }) if waiting_slack else None
        response = AnswerResponse.model_validate(snapshot.model_copy(update={"actions": [receipt]}).model_dump(mode="json"))
        status = "completed" if completed else "failed"
        log_event(logger, logging.INFO, "agent_response", session_id=session_id[:8],
                  request_id=request_id, answer_id=answer_id, outcome=status, action="answer_export",
                  artifact_id=receipt.artifact.artifact_id if receipt.artifact else None,
                  problem_code=receipt.error_code)
        return AnswerExportResult(
            status=status, answer_id=answer_id, request_id=request_id, response=response,
            message="TXT 파일을 저장했습니다. 아래에서 다운로드할 수 있습니다." if completed else
                    "파일을 만들지 못했습니다. 기존 답변은 유지됩니다. 같은 답변으로 다시 시도해 주세요.",
        )
