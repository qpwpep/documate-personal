"""Deterministic completion boundary for Streamlit's background chat receiver."""
from streamlit.testing.v1 import AppTest


def finish_chat_response(app: AppTest) -> AppTest:
    tasks = [conversation.chat_task for conversation in app.session_state["conversations"].values()
             if conversation.chat_task is not None]
    for task in tasks:
        assert task.finished.wait(2), "HTTP test response did not finish"
    if tasks:
        app.run()
    return app
