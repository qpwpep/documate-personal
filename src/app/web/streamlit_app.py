import logging
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from src.app.client import (
    AgentSessionClient,
    AnswerExportAPIError,
    UploadAPIError,
)
from src.app.web.streamlit_chat import process_chat_prompt, render_chat_history
from src.app.web.streamlit_browser_state import edit_question, render_browser_state, restore_browser_session, restore_composer
from src.app.web.streamlit_conversations import render_conversation_notice
from src.app.web.streamlit_requests import (
    ChatSubmission, SaveSubmission, UploadSubmission, discard_saved_prompt,
    queue_chat, queue_save, queue_upload, send_saved_prompt,
)
from src.app.web.streamlit_intro import render_intro
from src.app.web.streamlit_page import warn_if_utf8_mode_disabled_once
from src.app.web.streamlit_sidebar import render_sidebar
from src.app.web.streamlit_styles import configure_page
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
    conversation = get_conversation()
    if get_upload_manifest() is None and get_pending_upload() is None and not conversation.is_read_only:
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
        manifest=manifest, uploads_busy=get_pending_upload() is not None or conversation.is_read_only,
        interactions_disabled=busy,
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

    session_path = get_session_path()
    messages = get_messages()
    if (not messages and not busy and not conversation.is_read_only
            and not conversation.saved_prompt and pending_upload is None and manifest is not None):
        render_intro(DEFAULT_DOCS, has_uploads=bool(manifest.files))
    render_chat_history(
        messages, SETTINGS.fastapi_url, on_save=lambda answer_id: queue_save(conversation, answer_id),
        actions_disabled=busy or conversation.is_read_only,
        on_retry=(lambda prompt: queue_chat(conversation, prompt)) if not pending_upload and not conversation.saved_prompt else None,
        on_edit=(lambda prompt: edit_question(conversation, prompt)) if not pending_upload and not conversation.saved_prompt else None,
    )
    if conversation.is_read_only or queued is not None or pending_upload is not None or conversation.saved_prompt or manifest is None:
        # The native widget disappears on these screens; its next instance
        # needs the independently retained text, including after reconnection.
        conversation.draft.needs_restore = True
    if conversation.is_read_only:
        render_conversation_notice()
        return
    if queued is not None:
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

    saved_prompt = conversation.saved_prompt
    if saved_prompt:
        st.info(f"아직 보내지 않은 질문: {saved_prompt}")
        st.button("보류한 질문 보내기", key="documate_send_saved_upload_prompt",
                  on_click=send_saved_prompt, args=(conversation,))
        st.button("보류한 질문 지우기", key="documate_discard_saved_upload_prompt",
                  on_click=discard_saved_prompt, args=(conversation,))
        st.chat_input("보류한 질문을 보내거나 지운 뒤 새 질문을 입력하세요", disabled=True,
                      key=conversation.widget_key("held_input"))
        return

    suffixes = enabled_upload_suffixes(docling_enabled=getattr(SETTINGS, "docling_enabled", False))
    st.caption(f"{' · '.join(sorted(suffixes))} / 최대 {SETTINGS.upload_max_files}개 / 파일당 {SETTINGS.upload_max_file_mib} MiB / 전체 {SETTINGS.upload_max_total_mib} MiB")
    st.html('<p id="documate-upload-selection" role="status" aria-live="polite">'
            '선택한 파일은 전송 후 검색에 사용됩니다. 파일만 먼저 전송할 수도 있습니다.</p>')

    restore_composer(conversation)
    chat_submission = st.chat_input(
        "공식 문서나 업로드한 자료에 대해 질문하세요",
        accept_file="multiple",
        file_type=[suffix.lstrip(".") for suffix in sorted(suffixes)],
        max_upload_size=SETTINGS.upload_max_file_mib,
        key=conversation.widget_key("chat_input"),
    )
    typed_prompt, attached_files = _split_chat_submission(chat_submission)
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
    if not validate_conversation(conversation):
        if submission.kind == "chat":
            conversation.saved_prompt = submission.prompt
        elif submission.kind == "upload" and conversation.pending_upload is not None:
            conversation.pending_upload.requires_confirmation = True
        elif submission.kind == "save":
            for message in conversation.messages:
                if message["role"] == "assistant" and message["result"] is not None and message["result"].answer_id == submission.answer_id:
                    message["export_error"] = "저장 전에 서버 상태를 확인하지 못했습니다. 대화 연결을 확인한 뒤 같은 답변으로 다시 시도해 주세요."
        return
    conversation.operation_in_progress = True
    try:
        if submission.kind == "upload":
            with st.spinner("파일을 검증하고 함께 검색할 자료를 준비합니다…"):
                commit_pending_upload()
            return
        if submission.kind == "save":
            targets = [message for message in conversation.messages if message["role"] == "assistant"
                       and message["result"] is not None and message["result"].answer_id == submission.answer_id]
            try:
                with st.spinner("선택한 답변을 출처와 함께 저장합니다…"):
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
        process_chat_prompt(
            stream_agent=conversation.client.stream,
            prompt=submission.prompt,
            append_user_message=conversation.messages.append,
            append_assistant_message=conversation.messages.append,
        )
    finally:
        conversation.operation_in_progress = False
        conversation.draft.needs_restore = True


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
        pending.error = str(exc)
        if exc.files:
            pending.error += "\n" + "\n".join(f"{item.get('name', '파일')}: {item.get('message', item.get('code', '실패'))}" for item in exc.files)
        if exc.status_code == 409:
            pending.needs_refresh_review = True
            try:
                client.refresh_uploads()
            except UploadAPIError as refresh_error:
                pending.error += f"\n첨부 목록 새로고침 실패: {refresh_error}"
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
            pending.error = str(exc)
        return
    if not _is_current_pending(client, pending):
        return
    pending.files, pending.remove = review_upload_changes(manifest, files=pending.files, remove=pending.remove)
    pending.base_manifest = manifest.model_copy(deep=True)
    pending.request = None
    pending.replace_file_ids.clear()
    pending.error = None
    pending.needs_refresh_review = False
    pending.attempted = False


def _render_pending_upload() -> None:
    pending = get_pending_upload()
    if pending is None:
        return
    st.markdown('<p class="dm-upload-summary"><strong>첨부 변경 준비 중</strong></p>', unsafe_allow_html=True)
    for item in pending.files:
        st.markdown(f'<p class="dm-upload-help">{escape(item.name)} · {item.size_bytes / 1024:.1f} KiB · 반영 확인 대기</p>', unsafe_allow_html=True)
    if pending.remove:
        st.markdown('<p class="dm-upload-help">선택한 파일을 이후 검색에서 제외합니다.</p>', unsafe_allow_html=True)
    if pending.clear:
        st.markdown('<p class="dm-upload-help">모든 첨부를 이후 검색에서 제외합니다. 대화와 기존 인용은 유지됩니다.</p>', unsafe_allow_html=True)
    if pending.error:
        st.error(pending.error)
        st.info("질문은 보내지 않았습니다. 서버가 확인한 첨부 목록만 사용합니다.")
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
    cancel_label = "대기 화면 닫기" if pending.attempted or pending.had_failure else "첨부 준비 취소"
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
