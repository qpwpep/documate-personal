from __future__ import annotations

import unittest
from unittest.mock import patch

from src.app.web import streamlit_page


class _FakeStreamlit:
    def __init__(self) -> None:
        self.markdowns: list[tuple[str, bool]] = []
        self.query_params: dict[str, str] = {}
        self.session_state: dict[str, str] = {}

    def markdown(self, body: str, unsafe_allow_html: bool = False) -> None:
        self.markdowns.append((body, unsafe_allow_html))


class StreamlitPageTest(unittest.TestCase):
    def test_render_theme_styles_emits_light_override(self) -> None:
        fake_st = _FakeStreamlit()

        with patch.object(streamlit_page, "st", fake_st):
            streamlit_page.render_theme_styles("라이트")

        self.assertEqual(len(fake_st.markdowns), 1)
        body, unsafe = fake_st.markdowns[0]
        self.assertTrue(unsafe)
        self.assertIn("--dm-bg: #f7f5ef;", body)
        self.assertIn("--dm-text: #202124;", body)
        self.assertIn("--dm-chat-input-bg: #fffdfa;", body)
        self.assertIn("--dm-chat-attachment-bg: #f8f6f1;", body)
        self.assertIn("--dm-chat-attachment-text: #202124;", body)
        self.assertIn("--dm-chat-icon: #276f66;", body)
        self.assertIn("--dm-assistant-bg:", body)
        self.assertIn("--dm-inline-code-bg: rgba(39, 111, 102, 0.10);", body)

    def test_render_theme_styles_emits_dark_override(self) -> None:
        fake_st = _FakeStreamlit()

        with patch.object(streamlit_page, "st", fake_st):
            streamlit_page.render_theme_styles("다크")

        self.assertEqual(len(fake_st.markdowns), 1)
        body, unsafe = fake_st.markdowns[0]
        self.assertTrue(unsafe)
        self.assertIn("--dm-bg: #101214;", body)
        self.assertIn("--dm-text: #f5f1e8;", body)
        self.assertIn("--dm-chat-input-bg: #1d1e20;", body)
        self.assertIn("--dm-chat-attachment-bg: #242827;", body)
        self.assertIn("--dm-chat-attachment-text: #f5f1e8;", body)
        self.assertIn("--dm-chat-icon: #78d1c1;", body)
        self.assertIn("--dm-assistant-bg:", body)
        self.assertIn("--dm-inline-code-bg: rgba(120, 209, 193, 0.12);", body)

    def test_render_theme_styles_keeps_system_mode_css_media_query(self) -> None:
        fake_st = _FakeStreamlit()

        with patch.object(streamlit_page, "st", fake_st):
            streamlit_page.render_theme_styles("시스템")

        self.assertEqual(fake_st.markdowns, [])
        self.assertIn("@media (prefers-color-scheme: dark)", streamlit_page._APP_CSS)
        self.assertIn('[data-testid="stChatInput"] > div:focus-within', streamlit_page._APP_CSS)
        self.assertIn("caret-color: var(--dm-accent) !important;", streamlit_page._APP_CSS)
        self.assertIn("--dm-chat-attachment-text: #202124;", streamlit_page._APP_CSS)
        self.assertIn("--dm-chat-icon: #276f66;", streamlit_page._APP_CSS)
        self.assertIn('[data-testid="stChatInput"] button svg', streamlit_page._APP_CSS)
        self.assertIn(
            '[data-testid="stChatInput"] [data-testid="stChatInputFile"] > div:first-child',
            streamlit_page._APP_CSS,
        )
        self.assertIn(".dm-save-note", streamlit_page._APP_CSS)

    def test_sync_theme_from_query_params_sets_session_theme(self) -> None:
        fake_st = _FakeStreamlit()
        fake_st.query_params["theme"] = "dark"

        with patch.object(streamlit_page, "st", fake_st):
            streamlit_page._sync_theme_from_query_params()

        self.assertEqual(fake_st.session_state["documate_theme_mode"], "다크")

    def test_quick_prompts_are_sampled_once_per_session(self) -> None:
        fake_st = _FakeStreamlit()
        sampled_prompts = [
            "추천 1",
            "추천 2",
            "추천 3",
            "추천 4",
        ]

        with patch.object(streamlit_page, "st", fake_st), patch(
            "src.app.web.streamlit_page.random.sample",
            return_value=sampled_prompts,
        ) as mock_sample:
            first_prompts = streamlit_page._get_quick_prompts_for_session()
            second_prompts = streamlit_page._get_quick_prompts_for_session()

        self.assertEqual(first_prompts, sampled_prompts)
        self.assertEqual(second_prompts, sampled_prompts)
        mock_sample.assert_called_once_with(
            streamlit_page._QUICK_PROMPTS,
            streamlit_page._QUICK_PROMPT_COUNT,
        )


if __name__ == "__main__":
    unittest.main()
