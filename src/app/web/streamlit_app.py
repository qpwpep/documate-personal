import logging
from time import monotonic
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.runtime.scriptrunner import RerunException, StopException

from src.app.client import (
    AgentSessionClient,
    AnswerExportAPIError,
    UploadAPIError,
)
from src.app.web.streamlit_chat import render_chat_history
from src.app.web.chat_response_task import collect_chat_response, start_chat_response
from src.app.web.streamlit_browser_state import edit_question, render_browser_state, restore_browser_session, restore_composer
from src.app.web.streamlit_conversations import render_conversation_notice
from src.app.web.streamlit_requests import (
    ChatSubmission, SaveSubmission, UploadSubmission, continue_upload_question,
    discard_saved_prompt, edit_saved_prompt, get_upload_followup_prompt,
    interrupt_operation, queue_chat, queue_save, queue_upload, send_saved_prompt,
)
from src.app.web.streamlit_intro import render_intro
from src.app.web.streamlit_page import warn_if_utf8_mode_disabled_once
from src.app.web.streamlit_sidebar import render_sidebar
from src.app.web.streamlit_styles import configure_page, render_conversation_header
from src.app.web.streamlit_theme import THEME_STATE_KEY
from src.app.web.streamlit_state import (
    Conversation,
    PendingUpload,
    ensure_session_state,
    ensure_theme_state,
    get_messages,
    get_session_client,
    get_session_path,
    get_upload_manifest,
    get_pending_upload,
    get_conversation,
    list_conversations,
    reset_chat_session,
    set_pending_upload,
    switch_conversation,
    validate_conversation,
)
from src.app.uploads import (
    build_upload_sync_request,
    discard_staged_files,
    review_upload_changes,
)
from src.core.domain_docs import DEFAULT_DOCS
from src.core.upload_formats import enabled_upload_suffixes
from src.infra.logging_utils import configure_logging
from src.infra.runtime_encoding import ensure_utf8_stdio
from src.infra.settings import get_settings


ensure_utf8_stdio()
configure_logging()
logger = logging.getLogger(__name__)
SETTINGS = get_settings()


def main() -> None:
    ensure_theme_state()
    configure_page(st.session_state[THEME_STATE_KEY])
    warn_if_utf8_mode_disabled_once()
    if not restore_browser_session():
        return
    ensure_session_state(logger)
    for existing in list_conversations():
        collect_chat_response(existing)
    conversation = get_conversation()
    if (get_upload_manifest() is None and get_pending_upload() is None and not conversation.is_read_only
            and not conversation.is_busy):
        validate_conversation(conversation)
    if conversation.followup_prompt and not conversation.saved_prompt:
        queue_chat(conversation, conversation.followup_prompt)
        if conversation.pending_request is not None and conversation.pending_request.kind == "chat":
            conversation.followup_prompt = None
    queued = conversation.pending_request
    pending_upload = get_pending_upload()
    auto_upload = (
        not conversation.is_read_only and pending_upload is not None and not pending_upload.attempted
        and not pending_upload.requires_confirmation
        and not any(item.conflicting_file_id and item.conflicting_file_id not in pending_upload.replace_file_ids
                    for item in pending_upload.files)
    )
    busy = queued is not None or conversation.operation_in_progress or auto_upload
    manifest = get_upload_manifest()
    sidebar_inputs = render_sidebar(
        manifest=manifest, uploads_busy=busy or get_pending_upload() is not None or conversation.is_read_only,
        interactions_disabled=busy and conversation.chat_task is None,
    )
    if sidebar_inputs.new_chat_requested:
        reset_chat_session(logger)
        st.rerun()
    selected_conversation = sidebar_inputs.selected_conversation
    if selected_conversation is not None:
        switch_conversation(selected_conversation)
        st.rerun()
    conversation.client.context.slack_recipient = sidebar_inputs.slack_recipient
    render_browser_state(conversation)
    render_conversation_header(
        conversation.title, attachment_count=len(manifest.files) if manifest is not None else None,
        read_only=conversation.is_read_only,
        attachment_error=bool(pending_upload and pending_upload.error),
    )

    session_path = get_session_path()
    messages = get_messages()
    # Clear the previous welcome screen before a long synchronous request starts.
    intro = st.empty()
    intro.empty()
    if (not messages and not busy and not conversation.is_read_only
            and not conversation.saved_prompt and pending_upload is None and manifest is not None):
        with intro.container():
            render_intro(DEFAULT_DOCS, has_uploads=bool(manifest.files))
    render_chat_history(
        messages, SETTINGS.fastapi_url, on_save=lambda answer_id: queue_save(conversation, answer_id),
        actions_disabled=busy or conversation.is_read_only,
        on_retry=(lambda prompt: queue_chat(conversation, prompt)) if not pending_upload and not conversation.saved_prompt else None,
        on_edit=(lambda prompt: edit_question(conversation, prompt)) if not pending_upload and not conversation.saved_prompt else None,
    )
    if any(item.chat_task is not None for item in list_conversations()):
        _render_chat_progress(conversation.id)
    upload_status = st.empty()
    upload_status.empty()
    if conversation.is_read_only or queued is not None or pending_upload is not None or conversation.saved_prompt or manifest is None:
        # The native widget disappears on these screens; its next instance
        # needs the independently retained text, including after reconnection.
        conversation.draft.needs_restore = True
    if conversation.is_read_only:
        render_conversation_notice()
        return
    if queued is not None:
        if queued.kind != "chat":
            st.chat_input("처리 중입니다…", disabled=True, key=conversation.widget_key("busy_input"))
        _execute_submission(conversation, queued)
        st.rerun()

    if manifest is None:
        if get_pending_upload() is not None:
            _render_pending_upload()
            return
        st.error("첨부 목록을 불러오지 못했습니다.")
        if st.button("첨부 서버 다시 연결", key="documate_reconnect_uploads"):
            st.rerun()
        return

    if sidebar_inputs.refresh_uploads_requested:
        client = get_session_client()
        try:
            validate_conversation(conversation)
            st.rerun()
        except UploadAPIError as exc:
            st.error(str(exc))
            return

    if get_pending_upload() is None and (sidebar_inputs.remove_file_id or sidebar_inputs.clear_uploads_requested):
        set_pending_upload(PendingUpload(
            base_manifest=manifest,
            remove=[sidebar_inputs.remove_file_id] if sidebar_inputs.remove_file_id else [],
            clear=sidebar_inputs.clear_uploads_requested,
        ))

    if get_pending_upload() is not None:
        _render_pending_upload()
        return

    if manifest.files:
        with upload_status.container():
            with st.expander(f"첨부 {len(manifest.files)}개 · 검색 가능", expanded=False):
                st.caption(" · ".join(item.name for item in manifest.files))
                st.caption("실제 사용한 자료는 답변의 근거에서 확인하세요.")
            previous_prompt = get_upload_followup_prompt(conversation)
            if previous_prompt is not None:
                st.info("첨부한 자료로 이전 질문을 계속할 수 있습니다.")
                st.text(previous_prompt)
                st.button("이전 질문 계속하기", key="documate_continue_upload_question", type="primary",
                          on_click=continue_upload_question, args=(conversation,))
                st.caption("버튼을 누르면 위 질문을 보냅니다. 입력창에 작성 중인 질문은 유지됩니다.")

    saved_prompt = conversation.saved_prompt
    if saved_prompt:
        st.info(f"아직 보내지 않은 질문: {saved_prompt}")
        st.button("질문과 첨부 수정", key="documate_edit_saved_upload_prompt", type="primary",
                  on_click=edit_saved_prompt, args=(conversation,))
        st.caption("질문을 입력창으로 가져옵니다. 파일을 다시 선택하거나 질문을 수정한 뒤 전송하세요.")
        st.button("보류한 질문 보내기", key="documate_send_saved_upload_prompt",
                  on_click=send_saved_prompt, args=(conversation,))
        st.button("보류한 질문 지우기", key="documate_discard_saved_upload_prompt",
                  on_click=discard_saved_prompt, args=(conversation,))
        st.chat_input("질문과 첨부 수정에서 파일을 다시 선택할 수 있습니다", disabled=True,
                      key=conversation.widget_key("held_input"))
        return

    suffixes = enabled_upload_suffixes(docling_enabled=getattr(SETTINGS, "docling_enabled", False))
    with st.expander("첨부 형식과 용량 안내", expanded=False):
        st.caption(f"{' · '.join(sorted(suffixes))} / 최대 {SETTINGS.upload_max_files}개 / 파일당 {SETTINGS.upload_max_file_mib} MiB / 전체 {SETTINGS.upload_max_total_mib} MiB")
        st.caption("입력창의 +로 파일을 선택한 뒤 전송하세요. 파일만 먼저 전송할 수도 있습니다.")
    restore_composer(conversation)
    chat_submission = st.chat_input(
        "질문을 입력하세요",
        accept_file="multiple",
        file_type=[suffix.lstrip(".") for suffix in sorted(suffixes)],
        max_upload_size=SETTINGS.upload_max_file_mib,
        key=conversation.widget_key("chat_input"),
    )
    typed_prompt, attached_files = _split_chat_submission(chat_submission)
    if conversation.chat_task is not None:
        # The browser blocks submission while preserving ordinary typing.
        # A delayed widget event must retain the draft, never queue a request.
        if typed_prompt is not None:
            conversation.draft.edit(typed_prompt)
            st.rerun()
        return
    if chat_submission is not None:
        conversation.draft.consume()
    prompt = typed_prompt
    if attached_files:
        staged = _stage_files(attached_files, session_path)
        if staged.errors:
            for error in staged.errors:
                st.error(error)
            if prompt:
                conversation.saved_prompt = prompt
            return
        if staged.unchanged_names:
            st.info("이미 첨부된 동일 파일은 유지했습니다: " + ", ".join(staged.unchanged_names))
        if staged.files:
            set_pending_upload(PendingUpload(
                base_manifest=manifest,
                files=staged.files, prompt=prompt,
            ))
            st.rerun()
            return

    if prompt:
        queue_chat(conversation, prompt)
        st.rerun()


def _execute_submission(conversation: Conversation, submission: ChatSubmission | SaveSubmission | UploadSubmission) -> None:
    """A consumed submission cannot be replayed by a rerun or navigation."""
    conversation.pending_request = None
    conversation.active_submission = submission
    conversation.operation_message_count = len(conversation.messages)
    conversation.operation_in_progress = True
    try:
        valid = validate_conversation(conversation, expected_submission=submission)
        if conversation.active_submission is not submission:
            return
        if not valid:
            if submission.kind == "chat":
                conversation.saved_prompt = submission.prompt
            elif submission.kind == "upload" and conversation.pending_upload is not None:
                conversation.pending_upload.requires_confirmation = True
            elif submission.kind == "save":
                for message in conversation.messages:
                    if message["role"] == "assistant" and message["result"] is not None and message["result"].answer_id == submission.answer_id:
                        message["export_error"] = "저장 전에 서버 상태를 확인하지 못했습니다. 대화 연결을 확인한 뒤 같은 답변으로 다시 시도해 주세요."
            return
        if submission.kind == "upload":
            with st.spinner("파일을 검증하고 함께 검색할 자료를 준비합니다…", show_time=True):
                commit_pending_upload()
            return
        if submission.kind == "save":
            targets = [message for message in conversation.messages if message["role"] == "assistant"
                       and message["result"] is not None and message["result"].answer_id == submission.answer_id]
            try:
                with st.spinner("선택한 답변을 출처와 함께 저장합니다…", show_time=True):
                    result = conversation.client.save_answer(submission.answer_id)
                for message in targets:
                    message["export_result"] = result
                    message.pop("export_error", None)
            except AnswerExportAPIError as exc:
                for message in targets:
                    message["export_error"] = str(exc)
                if exc.status_code == 410:
                    conversation.expired = True
            return
        start_chat_response(conversation, submission.prompt, submission.operation_id)
    except (RerunException, StopException):
        # A browser-triggered rerun stops receiving the response, not the
        # server operation. Never replay an utterance or its delivery actions.
        interrupt_operation(conversation, submission.operation_id)
        raise
    finally:
        if conversation.active_submission is submission and conversation.chat_task is None:
            conversation.active_submission = None
            conversation.operation_in_progress = False
            conversation.draft.needs_restore = True


@st.fragment(run_every=0.5)
def _render_chat_progress(conversation_id: str) -> None:
    completed = False
    for conversation in list_conversations():
        completed = collect_chat_response(conversation) or completed
    if completed:
        st.rerun()
    conversation = st.session_state["conversations"][conversation_id]
    task = conversation.chat_task
    if task is None:
        st.caption("다른 대화의 답변을 준비하고 있습니다. 대화 목록에서 돌아가 확인할 수 있습니다.")
        return
    with st.chat_message("assistant"):
        elapsed = int(monotonic() - task.started_at)
        st.markdown(f"**{task.progress}** · {elapsed}초")
        st.caption("기다리는 동안 다음 질문을 작성하거나 다른 대화를 볼 수 있습니다. 전송과 첨부 변경은 답변이 끝나면 다시 사용할 수 있습니다.")
        if elapsed >= 30:
            st.info("응답을 기다리는 시간이 길어지고 있습니다. 다른 대화를 보는 동안에도 처리는 계속됩니다.")
        if st.button("응답 수신 중단", key=conversation.widget_key("stop_response"),
                     help="화면의 응답 수신만 중단합니다. 서버 작업은 계속될 수 있습니다."):
            interrupt_operation(conversation, task.operation_id)
            st.rerun()
        st.caption("‘응답 수신 중단’은 서버 작업을 취소하지 않습니다.")


def _stage_files(files: list[Any], session_path: Path):
    return get_session_client().stage_files(
        files, session_path,
        max_files=SETTINGS.upload_max_files, max_file_mib=SETTINGS.upload_max_file_mib,
        max_total_mib=SETTINGS.upload_max_total_mib,
    )


def commit_pending_upload() -> bool:
    """Apply one prepared operation; retries preserve its identity and never send a question."""
    pending = get_pending_upload()
    if pending is None:
        return False
    if pending.needs_refresh_review:
        return False
    if pending.request is None:
        if any(item.conflicting_file_id and item.conflicting_file_id not in pending.replace_file_ids for item in pending.files):
            return False
        pending.request = build_upload_sync_request(
            pending.base_manifest, files=pending.files,
            replace_file_ids=pending.replace_file_ids, remove=pending.remove, clear=pending.clear,
        )
    pending.attempted = True
    client = get_session_client()
    session_path = get_session_path()
    try:
        client.sync_uploads(pending.request)
    except UploadAPIError as exc:
        if not _is_current_pending(client, pending):
            return False
        pending.had_failure = True
        pending.error = _upload_error_message(exc, pending)
        pending.error_detail = str(exc)
        if exc.files:
            pending.error_detail += "\n" + "\n".join(f"{item.get('name', '파일')}: {item.get('message', item.get('code', '실패'))}" for item in exc.files)
        if exc.status_code == 409:
            pending.needs_refresh_review = True
            try:
                client.refresh_uploads()
            except UploadAPIError as refresh_error:
                pending.error_detail += f"\n첨부 목록 새로고침 실패: {refresh_error}"
        return False
    if not _is_current_pending(client, pending):
        return False
    if pending.prompt:
        if pending.had_failure:
            get_conversation().saved_prompt = pending.prompt
        else:
            get_conversation().followup_prompt = pending.prompt
    discard_staged_files(pending.files, session_path)
    set_pending_upload(None)
    return True


def _upload_error_message(error: UploadAPIError, pending: PendingUpload) -> str:
    if error.code == "UPLOAD_VALIDATION_FAILED":
        message = "파일 내용을 읽지 못해 첨부 변경을 반영하지 않았습니다. 올바른 파일로 교체해 주세요."
        if any(Path(item.name).suffix.lower() == ".ipynb" for item in pending.files):
            message += " Notebook(.ipynb)은 Jupyter에서 열리는지 확인한 뒤 다시 저장해 주세요."
        return message
    if error.status_code == 413:
        return "첨부 용량 또는 개수 한도를 초과했습니다. 파일 크기와 개수를 줄인 뒤 다시 전송해 주세요."
    if error.status_code == 422:
        return "파일 형식이나 내용을 확인하지 못해 첨부 변경을 반영하지 않았습니다. 파일을 확인하거나 교체해 주세요."
    if error.status_code == 409:
        return "첨부 목록이 변경되어 이번 변경을 반영하지 않았습니다. 최신 목록으로 다시 적용하거나 첨부를 수정해 주세요."
    return "첨부 변경 결과를 확인하지 못했습니다. 같은 변경을 다시 시도하거나, 현재 첨부 목록을 확인한 뒤 파일을 다시 선택해 주세요."


def _is_current_pending(client: AgentSessionClient, pending: PendingUpload) -> bool:
    return get_session_client() is client and get_pending_upload() is pending


def _review_pending_again() -> None:
    pending = get_pending_upload()
    if pending is None:
        return
    client = get_session_client()
    try:
        manifest = client.refresh_uploads()
    except UploadAPIError as exc:
        if _is_current_pending(client, pending):
            pending.error = "최신 첨부 목록을 확인하지 못했습니다. 잠시 후 다시 시도하거나 첨부를 수정해 주세요."
            pending.error_detail = str(exc)
        return
    if not _is_current_pending(client, pending):
        return
    pending.files, pending.remove = review_upload_changes(manifest, files=pending.files, remove=pending.remove)
    pending.base_manifest = manifest.model_copy(deep=True)
    pending.request = None
    pending.replace_file_ids.clear()
    pending.error = None
    pending.error_detail = None
    pending.needs_refresh_review = False
    pending.attempted = False


def _edit_pending_upload() -> None:
    conversation = get_conversation()
    pending = conversation.pending_upload
    if pending is None or conversation.is_busy or conversation.is_read_only:
        return
    conversation.draft.edit(pending.prompt or conversation.saved_prompt or conversation.draft.text)
    conversation.saved_prompt = None
    conversation.pending_upload = None
    # An attempted upload may still be using its staging files on the server.
    # Leave those to session cleanup, just as cancellation does.
    if not pending.attempted and not pending.had_failure:
        discard_staged_files(pending.files, get_session_path())
    validate_conversation(conversation)


def _render_pending_upload() -> None:
    pending = get_pending_upload()
    if pending is None:
        return
    title = "첨부 변경을 완료하지 못했습니다" if pending.error else "첨부 변경 준비 중"
    file_status = "변경 미완료" if pending.error else "반영 확인 대기"
    error_attributes = ""
    if pending.error:
        operation_id = pending.request.operation_id if pending.request is not None else pending.error
        error_key = escape(f"{get_conversation().id}:{operation_id}", quote=True)
        error_attributes = f' tabindex="-1" data-dm-upload-error="{error_key}"'
    st.markdown(f'<p class="dm-upload-summary"{error_attributes}><strong>{title}</strong></p>', unsafe_allow_html=True)
    for item in pending.files:
        st.markdown(f'<p class="dm-upload-help">{escape(item.name)} · {item.size_bytes / 1024:.1f} KiB · {file_status}</p>', unsafe_allow_html=True)
    if pending.remove:
        st.markdown('<p class="dm-upload-help">선택한 파일을 이후 검색에서 제외합니다.</p>', unsafe_allow_html=True)
    if pending.clear:
        st.markdown('<p class="dm-upload-help">모든 첨부를 이후 검색에서 제외합니다. 대화와 기존 인용은 유지됩니다.</p>', unsafe_allow_html=True)
    if pending.error:
        st.error(pending.error)
        if pending.prompt:
            st.info(f"질문은 아직 보내지 않았습니다: {pending.prompt}")
        st.button("파일 다시 선택", key="documate_edit_pending_upload", type="primary", on_click=_edit_pending_upload)
        st.caption("입력창으로 돌아간 뒤 +로 올바른 파일을 선택하세요. 작성한 질문은 유지됩니다.")
        if pending.error_detail:
            with st.expander("오류 상세", expanded=False):
                st.text(pending.error_detail)
    conflicts = [item for item in pending.files if item.conflicting_file_id and item.conflicting_file_id not in pending.replace_file_ids]
    if pending.needs_refresh_review:
        if st.button("최신 첨부 목록으로 다시 적용", key="documate_review_uploads"):
            _review_pending_again()
            st.rerun()
    elif conflicts:
        st.warning("같은 이름의 다른 내용이 있습니다: " + ", ".join(item.name for item in conflicts))
        if st.button("같은 이름의 파일 교체", key="documate_confirm_upload_replacements"):
            pending.replace_file_ids.update(item.conflicting_file_id for item in conflicts)
            queue_upload(get_conversation())
            st.rerun()
    elif not pending.attempted and not pending.requires_confirmation:
        queue_upload(get_conversation())
        st.rerun()
    elif st.button("첨부 변경 다시 시도" if pending.attempted else "첨부 변경 적용", key="documate_retry_uploads"):
        queue_upload(get_conversation())
        st.rerun()
    if pending.error:
        continue_label = "기존 자료로 계속" if pending.base_manifest.files else "첨부 없이 계속"
        st.button(continue_label, key="documate_cancel_pending_upload", on_click=_edit_pending_upload)
        st.caption("이번 첨부 변경을 취소하고 서버의 자료 목록을 확인합니다. 질문은 자동으로 보내지 않습니다.")
        st.chat_input("첨부 복구가 필요합니다", disabled=True,
                      key=get_conversation().widget_key("failed_upload_input"))
        return
    cancel_label = "첨부 변경 취소" if pending.attempted or pending.had_failure else "첨부 준비 취소"
    if st.button(cancel_label, key="documate_cancel_pending_upload"):
        if pending.prompt:
            get_conversation().saved_prompt = pending.prompt
        if not pending.attempted and not pending.had_failure:
            discard_staged_files(pending.files, get_session_path())
        set_pending_upload(None)
        client = get_session_client()
        try:
            client.refresh_uploads()
        except UploadAPIError:
            pass
        st.rerun()


def _split_chat_submission(submission: Any) -> tuple[str | None, list[Any]]:
    if submission is None:
        return None, []
    if isinstance(submission, str):
        text, files = submission, []
    else:
        text = str(getattr(submission, "text", "") or "")
        files = getattr(submission, "files", None) or []
    return text if text.strip() else None, list(files)


if __name__ == "__main__":
    main()
