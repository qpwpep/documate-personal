from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.web import streamlit_state


class StreamlitStateTest(unittest.TestCase):
    def test_ensure_session_state_initializes_defaults_and_session_path(self) -> None:
        fake_st = SimpleNamespace(session_state={})
        with TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                with patch.object(streamlit_state, "st", fake_st), patch(
                    "src.web.streamlit_state.uuid.uuid4",
                    return_value="session-123",
                ), patch("src.web.streamlit_state.log_event") as mock_log_event:
                    streamlit_state.ensure_session_state(streamlit_state.logging.getLogger(__name__))

                    self.assertEqual(streamlit_state.get_session_id(), "session-123")
                    self.assertIsNone(streamlit_state.get_uploaded_file_name())
                    self.assertEqual(len(streamlit_state.get_messages()), 1)
                    self.assertEqual(streamlit_state.get_messages()[0]["role"], "assistant")
                    self.assertTrue((Path(temp_dir) / "uploads" / "session-123").exists())
                    mock_log_event.assert_called_once()
            finally:
                os.chdir(original_cwd)

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


if __name__ == "__main__":
    unittest.main()
