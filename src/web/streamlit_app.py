import logging
import streamlit as st

from src.domain_docs import DEFAULT_DOCS
from src.logging_utils import configure_logging
from src.runtime_encoding import ensure_utf8_stdio
from src.settings import get_settings
from src.web.streamlit_api_client import AgentRequestContext, get_agent_response
from src.web.streamlit_chat import process_chat_prompt, render_chat_history
from src.web.streamlit_page import (
    configure_page,
    render_intro,
    render_sidebar,
    warn_if_utf8_mode_disabled_once,
)
from src.web.streamlit_state import (
    append_message,
    clear_uploaded_file_name,
    ensure_session_state,
    get_messages,
    get_session_id,
    get_session_path,
    get_uploaded_file_name,
    set_uploaded_file_name,
)
from src.web.streamlit_upload_handler import sync_uploaded_file


ensure_utf8_stdio()
configure_logging()
logger = logging.getLogger(__name__)
SETTINGS = get_settings()


def main() -> None:
    configure_page()
    warn_if_utf8_mode_disabled_once()
    sidebar_inputs = render_sidebar()
    ensure_session_state(logger)
    session_path = get_session_path()

    render_intro(DEFAULT_DOCS)
    render_chat_history(get_messages(), SETTINGS.fastapi_url)

    prompt = st.chat_input("여기에 질문을 입력하세요...")
    if prompt:
        def call_agent(user_input: str):
            upload_file_name = get_uploaded_file_name()
            upload_file_path = (
                (session_path / upload_file_name).as_posix()
                if upload_file_name
                else None
            )
            return get_agent_response(
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
            call_agent=call_agent,
            prompt=prompt,
            append_user_message=append_message,
            append_assistant_message=append_message,
        )

    uploaded_file = st.file_uploader(
        label="파일 업로드 (.py, .ipynb 등 챗봇에게 질문할 때 사용할 파일을 업로드 하세요.)",
        type=["ipynb", "py"],
        width=450,
    )
    sync_result = sync_uploaded_file(
        uploaded_file=uploaded_file,
        session_path=session_path,
        current_file_name=get_uploaded_file_name(),
    )
    if sync_result.error_message:
        clear_uploaded_file_name()
        st.error(sync_result.error_message)
    elif sync_result.changed:
        if sync_result.file_name:
            set_uploaded_file_name(sync_result.file_name)
        else:
            clear_uploaded_file_name()


main()
