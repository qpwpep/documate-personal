from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.app.client import AgentRequestContext, AgentSessionClient
from src.app.web import streamlit_state
from src.core.uploads import UploadManifest


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
            ):
                streamlit_state.ensure_session_state(streamlit_state.logging.getLogger(__name__))

                self.assertEqual(streamlit_state.get_session_id(), "session-123")
                self.assertIsNone(streamlit_state.get_upload_manifest())
                client = streamlit_state.get_session_client()
                self.assertEqual(client.context.session_id, "session-123")
                streamlit_state.ensure_session_state(streamlit_state.logging.getLogger(__name__))
                self.assertIs(streamlit_state.get_session_client(), client)
                self.assertNotIn("upload_manifest", fake_st.session_state)
                self.assertEqual(len(streamlit_state.get_messages()), 1)
                self.assertEqual(streamlit_state.get_messages()[0]["role"], "assistant")
                self.assertTrue((uploads_dir / "session-123").exists())

    def test_reset_chat_session_starts_clean_conversation(self) -> None:
        old_client = AgentSessionClient(
            AgentRequestContext(fastapi_url="http://test", session_id="old-session"),
            manifest=UploadManifest(epoch="old-epoch", revision=2, files=[]),
        )
        fake_st = SimpleNamespace(
            session_state={
                "session_id": "old-session",
                "session_client": old_client,
                "pending_upload": streamlit_state.PendingUpload(base_manifest=old_client.manifest, clear=True),
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
            ):
                streamlit_state.reset_chat_session(streamlit_state.logging.getLogger(__name__))

                self.assertEqual(streamlit_state.get_session_id(), "new-session")
                self.assertIsNone(streamlit_state.get_upload_manifest())
                self.assertIsNot(streamlit_state.get_session_client(), old_client)
                self.assertEqual(streamlit_state.get_session_client().context.session_id, "new-session")
                self.assertIsNone(streamlit_state.get_pending_upload())
                old_client.manifest = UploadManifest(epoch="late-old-result", revision=3, files=[])
                self.assertIsNone(streamlit_state.get_upload_manifest())
                self.assertNotIn("documate_quick_prompts", fake_st.session_state)
                self.assertNotIn("upload_saved_prompt", fake_st.session_state)
                self.assertNotIn("upload_followup_prompt", fake_st.session_state)
                self.assertEqual(len(streamlit_state.get_messages()), 1)
                self.assertEqual(streamlit_state.get_messages()[0]["role"], "assistant")
                self.assertTrue((uploads_dir / "new-session").exists())


if __name__ == "__main__":
    unittest.main()
