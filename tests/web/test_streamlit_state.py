from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.app.web import streamlit_state


class StreamlitStateTest(unittest.TestCase):
    def test_ensure_session_state_initializes_defaults_and_session_path(self) -> None:
        fake_st = SimpleNamespace(session_state={})
        with TemporaryDirectory() as temp_dir:
            uploads_dir = Path(temp_dir) / "uploads"
            with patch.object(streamlit_state, "st", fake_st), patch(
                "src.app.web.streamlit_state.get_uploads_dir",
                return_value=uploads_dir,
            ), patch(
                "src.app.web.streamlit_state.uuid.uuid4",
                return_value="session-123",
            ), patch("src.app.web.streamlit_state.log_event") as mock_log_event:
                streamlit_state.ensure_session_state(streamlit_state.logging.getLogger(__name__))

                self.assertEqual(streamlit_state.get_session_id(), "session-123")
                self.assertIsNone(streamlit_state.get_uploaded_file_name())
                self.assertEqual(len(streamlit_state.get_messages()), 1)
                self.assertEqual(streamlit_state.get_messages()[0]["role"], "assistant")
                self.assertTrue((uploads_dir / "session-123").exists())
                mock_log_event.assert_called_once()

    def test_uploaded_file_name_helpers_and_append_message(self) -> None:
        fake_st = SimpleNamespace(
            session_state={
                "session_id": "session-123",
                "uploaded_file_name": None,
                "messages": [],
            }
        )

        with patch.object(streamlit_state, "st", fake_st):
            streamlit_state.set_uploaded_file_name("sample.py")
            self.assertEqual(streamlit_state.get_uploaded_file_name(), "sample.py")

            streamlit_state.append_message(
                {
                    "role": "assistant",
                    "content": "hello",
                    "file_path": "",
                    "evidence": [],
                }
            )
            self.assertEqual(len(streamlit_state.get_messages()), 1)

            streamlit_state.clear_uploaded_file_name()
            self.assertIsNone(streamlit_state.get_uploaded_file_name())

    def test_reset_chat_session_starts_clean_conversation(self) -> None:
        fake_st = SimpleNamespace(
            session_state={
                "session_id": "old-session",
                "uploaded_file_name": "sample.py",
                "documate_quick_prompts": ["old prompt"],
                "messages": [
                    {
                        "role": "user",
                        "content": "previous",
                        "file_path": "",
                        "evidence": [],
                    }
                ],
            }
        )
        with TemporaryDirectory() as temp_dir:
            uploads_dir = Path(temp_dir) / "uploads"
            with patch.object(streamlit_state, "st", fake_st), patch(
                "src.app.web.streamlit_state.get_uploads_dir",
                return_value=uploads_dir,
            ), patch(
                "src.app.web.streamlit_state.uuid.uuid4",
                return_value="new-session",
            ), patch("src.app.web.streamlit_state.log_event") as mock_log_event:
                streamlit_state.reset_chat_session(streamlit_state.logging.getLogger(__name__))

                self.assertEqual(streamlit_state.get_session_id(), "new-session")
                self.assertIsNone(streamlit_state.get_uploaded_file_name())
                self.assertNotIn("documate_quick_prompts", fake_st.session_state)
                self.assertEqual(len(streamlit_state.get_messages()), 1)
                self.assertEqual(streamlit_state.get_messages()[0]["role"], "assistant")
                self.assertTrue((uploads_dir / "new-session").exists())
                mock_log_event.assert_called_once()


if __name__ == "__main__":
    unittest.main()
