from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.app.web import streamlit_state
from tests.web.answer_fixtures import answer_response


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
                    "response": answer_response("hello"),
                }
            )
            self.assertEqual(len(streamlit_state.get_messages()), 1)
            self.assertEqual(streamlit_state.get_messages()[0]["response"], answer_response("hello"))

            streamlit_state.clear_uploaded_file_name()
            self.assertIsNone(streamlit_state.get_uploaded_file_name())

    def test_reset_chat_session_starts_clean_conversation(self) -> None:
        fake_st = SimpleNamespace(
            session_state={
                "session_id": "old-session",
                "uploaded_file_name": "sample.py",
                "documate_quick_prompts": ["old prompt"],
                "upload_saved_prompt": "old held question",
                "upload_followup_prompt": "old queued question",
                "messages": [
                    {
                        "role": "user",
                        "content": "previous",
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
                self.assertNotIn("upload_saved_prompt", fake_st.session_state)
                self.assertNotIn("upload_followup_prompt", fake_st.session_state)
                self.assertEqual(len(streamlit_state.get_messages()), 1)
                self.assertEqual(streamlit_state.get_messages()[0]["role"], "assistant")
                self.assertTrue((uploads_dir / "new-session").exists())
                mock_log_event.assert_called_once()


if __name__ == "__main__":
    unittest.main()


def test_confirmed_upload_state_changes_without_erasing_conversation(monkeypatch):
    """Attachment changes replace the confirmed manifest while preserving past answers."""
    from src.core.uploads import UploadManifest
    fake_st = SimpleNamespace(session_state={"messages": [{"role": "user", "content": "keep"}]})
    monkeypatch.setattr(streamlit_state, "st", fake_st)
    manifest = UploadManifest(epoch="epoch-one", revision=2, files=[])
    streamlit_state.set_upload_manifest(manifest)
    assert streamlit_state.get_upload_manifest() == manifest
    assert streamlit_state.get_messages() == [{"role": "user", "content": "keep"}]


def test_pending_upload_keeps_operation_id_across_reruns(monkeypatch):
    """A retry retains the same operation identity while confirmed files stay unchanged."""
    from src.app.web.streamlit_upload_handler import PendingUploadOperation
    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(streamlit_state, "st", fake_st)
    pending = PendingUploadOperation(epoch="epoch-one", expected_revision=1)
    streamlit_state.set_pending_upload(pending)
    assert streamlit_state.get_pending_upload().operation_id == pending.operation_id
    streamlit_state.set_pending_upload(None)
    assert streamlit_state.get_pending_upload() is None


def test_invalidating_upload_confirmation_preserves_conversation(monkeypatch):
    """An uncertain server result requires a fresh manifest without erasing visible answers."""
    from src.core.uploads import UploadManifest
    messages = [{"role": "user", "content": "keep"}]
    fake_st = SimpleNamespace(session_state={
        "messages": messages, "upload_manifest": UploadManifest(epoch="old-epoch", revision=1),
    })
    monkeypatch.setattr(streamlit_state, "st", fake_st)

    streamlit_state.set_upload_manifest(None)

    assert streamlit_state.get_upload_manifest() is None
    assert streamlit_state.get_messages() == messages
