import logging
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from src.app.client import (
    AgentSessionClient,
    UploadAPIError,
)
from src.app.web.streamlit_chat import process_chat_prompt, render_chat_history
from src.app.web.streamlit_intro import render_intro
from src.app.web.streamlit_page import warn_if_utf8_mode_disabled_once
from src.app.web.streamlit_sidebar import render_sidebar
from src.app.web.streamlit_styles import configure_page
from src.app.web.streamlit_theme import render_theme_styles
from src.app.web.streamlit_state import (
    append_message,
    PendingUpload,
    ensure_session_state,
    get_messages,
    get_session_client,
    get_session_path,
    get_upload_manifest,
    get_pending_upload,
    reset_chat_session,
    set_pending_upload,
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
    configure_page()
    warn_if_utf8_mode_disabled_once()
    ensure_session_state(logger)

    manifest_error = None
    if get_upload_manifest() is None and get_pending_upload() is None:
        try:
            get_session_client().refresh_uploads()
        except UploadAPIError as exc:
            manifest_error = str(exc)

    manifest = get_upload_manifest()
    sidebar_inputs = render_sidebar(manifest=manifest, uploads_busy=get_pending_upload() is not None)
    if sidebar_inputs.new_chat_requested:
        reset_chat_session(logger)
        st.rerun()

    render_theme_styles(sidebar_inputs.theme_mode)

    session_path = get_session_path()
    messages = get_messages()
    selected_prompt = render_intro(DEFAULT_DOCS) if len(messages) <= 1 else None
    render_chat_history(messages, SETTINGS.fastapi_url)

    if manifest is None:
        if get_pending_upload() is not None:
            _render_pending_upload()
            return
        st.error(manifest_error or "첨부 목록을 불러오지 못했습니다.")
        if st.button("첨부 서버 다시 연결", key="documate_reconnect_uploads"):
            st.rerun()
        return

    if sidebar_inputs.refresh_uploads_requested:
        client = get_session_client()
        try:
            client.refresh_uploads()
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

    saved_prompt = st.session_state.get("upload_saved_prompt")
    send_saved_prompt = False
    if saved_prompt:
        st.info(f"아직 보내지 않은 질문: {saved_prompt}")
        send_saved_prompt = st.button("보류한 질문 보내기", key="documate_send_saved_upload_prompt")
        if st.button("보류한 질문 지우기", key="documate_discard_saved_upload_prompt"):
            st.session_state.pop("upload_saved_prompt", None)
            st.rerun()

    suffixes = enabled_upload_suffixes(docling_enabled=getattr(SETTINGS, "docling_enabled", False))
    st.caption(f"{' · '.join(sorted(suffixes))} / 최대 {SETTINGS.upload_max_files}개 / 파일당 {SETTINGS.upload_max_file_mib} MiB / 전체 {SETTINGS.upload_max_total_mib} MiB")

    chat_submission = st.chat_input(
        "공식 문서나 업로드한 자료에 대해 질문하세요",
        accept_file="multiple",
        file_type=[suffix.lstrip(".") for suffix in sorted(suffixes)],
        max_upload_size=SETTINGS.upload_max_file_mib,
    )
    typed_prompt, attached_files = _split_chat_submission(chat_submission)
    prompt = typed_prompt or st.session_state.pop("upload_followup_prompt", None) or (saved_prompt if send_saved_prompt else None) or selected_prompt
    if send_saved_prompt:
        st.session_state.pop("upload_saved_prompt", None)
    if attached_files:
        staged = _stage_files(attached_files, session_path)
        if staged.errors:
            for error in staged.errors:
                st.error(error)
            if prompt:
                st.session_state["upload_saved_prompt"] = prompt
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
        if sidebar_inputs.slack_recipient_error is not None:
            st.error(sidebar_inputs.slack_recipient_error)
            st.session_state["upload_saved_prompt"] = prompt
            return

        client = get_session_client()
        client.context.slack_recipient = sidebar_inputs.slack_recipient

        def append_current_message(message):
            if get_session_client() is client:
                append_message(message)

        process_chat_prompt(
            stream_agent=client.stream,
            prompt=prompt,
            append_user_message=append_current_message,
            append_assistant_message=append_current_message,
        )


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
        key = "upload_saved_prompt" if pending.had_failure else "upload_followup_prompt"
        st.session_state[key] = pending.prompt
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
            with st.spinner("파일을 검증하고 함께 검색할 자료를 준비합니다…"):
                commit_pending_upload()
            st.rerun()
    elif not pending.attempted:
        with st.spinner("파일을 검증하고 함께 검색할 자료를 준비합니다…"):
            commit_pending_upload()
        st.rerun()
    elif st.button("첨부 변경 다시 시도", key="documate_retry_uploads"):
        with st.spinner("동일한 첨부 변경의 처리 결과를 확인합니다…"):
            commit_pending_upload()
        st.rerun()
    cancel_label = "대기 화면 닫기" if pending.attempted or pending.had_failure else "첨부 준비 취소"
    if st.button(cancel_label, key="documate_cancel_pending_upload"):
        if pending.prompt:
            st.session_state["upload_saved_prompt"] = pending.prompt
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
        return submission.strip() or None, []

    text = str(getattr(submission, "text", "") or "").strip() or None
    files = getattr(submission, "files", None) or []
    return text, list(files)


if __name__ == "__main__":
    main()
