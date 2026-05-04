import logging
from typing import Any

import streamlit as st

from src.app.web.streamlit_api_client import AgentRequestContext, stream_agent_response
from src.app.web.streamlit_chat import process_chat_prompt, render_chat_history
from src.app.web.streamlit_page import (
    configure_page,
    render_intro,
    render_sidebar,
    render_theme_styles,
    warn_if_utf8_mode_disabled_once,
)
from src.app.web.streamlit_state import (
    append_message,
    clear_uploaded_file_name,
    ensure_session_state,
    get_messages,
    get_session_id,
    get_session_path,
    get_uploaded_file_name,
    reset_chat_session,
    set_uploaded_file_name,
)
from src.app.web.streamlit_upload_handler import sync_uploaded_file
from src.core.domain_docs import DEFAULT_DOCS
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

    sidebar_inputs = render_sidebar(get_uploaded_file_name())
    if sidebar_inputs.new_chat_requested:
        reset_chat_session(logger)
        st.rerun()

    render_theme_styles(sidebar_inputs.theme_mode)

    session_path = get_session_path()
    messages = get_messages()
    selected_prompt = render_intro(DEFAULT_DOCS) if len(messages) <= 1 else None
    render_chat_history(messages, SETTINGS.fastapi_url)

    chat_submission = st.chat_input(
        "공식 문서나 업로드한 코드에 대해 질문하세요",
        accept_file=True,
        file_type=["py", "ipynb"],
    )
    typed_prompt, attached_file = _split_chat_submission(chat_submission)
    prompt = typed_prompt or selected_prompt
    if attached_file is not None:
        sync_result = sync_uploaded_file(
            uploaded_file=attached_file,
            session_path=session_path,
            current_file_name=get_uploaded_file_name(),
        )
        if sync_result.error_message:
            clear_uploaded_file_name()
            st.error(sync_result.error_message)
            return
        if sync_result.changed:
            if sync_result.file_name:
                set_uploaded_file_name(sync_result.file_name)
            else:
                clear_uploaded_file_name()
        if not prompt:
            st.rerun()
            return

    if prompt:

        def stream_agent(user_input: str):
            upload_file_name = get_uploaded_file_name()
            upload_file_path = (
                (session_path / upload_file_name).as_posix()
                if upload_file_name
                else None
            )
            return stream_agent_response(
                user_input,
                AgentRequestContext(
                    fastapi_url=SETTINGS.fastapi_url,
                    session_id=get_session_id(),
                    slack_user_id=sidebar_inputs.slack_user_id,
                    slack_email=sidebar_inputs.slack_email,
                    slack_channel_id=sidebar_inputs.slack_channel_id,
                    upload_file_path=upload_file_path,
                ),
            )

        process_chat_prompt(
            stream_agent=stream_agent,
            prompt=prompt,
            append_user_message=append_message,
            append_assistant_message=append_message,
        )


def _split_chat_submission(submission: Any) -> tuple[str | None, Any | None]:
    if submission is None:
        return None, None
    if isinstance(submission, str):
        return submission.strip() or None, None

    text = str(getattr(submission, "text", "") or "").strip() or None
    files = getattr(submission, "files", None) or []
    attached_file = files[0] if files else None
    return text, attached_file


main()
