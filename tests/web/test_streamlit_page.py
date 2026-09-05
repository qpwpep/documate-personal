from __future__ import annotations

import unittest
from contextlib import nullcontext
from unittest.mock import patch

from src.app.web import streamlit_intro, streamlit_sidebar, streamlit_styles, streamlit_theme


class _FakeStreamlit:
    def __init__(self) -> None:
        self.markdowns: list[tuple[str, bool]] = []
        self.query_params: dict[str, str] = {}
        self.session_state: dict[str, object] = {}
        self.button_labels: list[str] = []
        self.sidebar = nullcontext()

    def set_page_config(self, **kwargs) -> None:
        pass

    def markdown(self, body: str, unsafe_allow_html: bool = False) -> None:
        self.markdowns.append((body, unsafe_allow_html))

    def columns(self, count: int):
        return [nullcontext() for _ in range(count)]

    def button(self, label: str, **kwargs) -> bool:
        self.button_labels.append(label)
        return False

    def radio(self, label: str, *, options, index: int, key: str, **kwargs):
        return self.session_state.get(key, options[index])

    def text_input(self, label: str, *, value: str, **kwargs) -> str:
        return value


class StreamlitPageTest(unittest.TestCase):
    def test_render_theme_styles_emits_light_override(self) -> None:
        fake_st = _FakeStreamlit()

        with patch.object(streamlit_theme, "st", fake_st):
            streamlit_theme.render_theme_styles("라이트")

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

        with patch.object(streamlit_theme, "st", fake_st):
            streamlit_theme.render_theme_styles("다크")

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

        with patch.object(streamlit_styles, "st", fake_st), patch.object(
            streamlit_theme, "st", fake_st
        ):
            streamlit_styles.configure_page()
            base_markdowns = list(fake_st.markdowns)
            streamlit_theme.render_theme_styles("시스템")

        self.assertEqual(fake_st.markdowns, base_markdowns)
        rendered_page = "\n".join(body for body, _ in fake_st.markdowns)
        self.assertIn("@media (prefers-color-scheme: dark)", rendered_page)
        self.assertIn('[data-testid="stChatInput"] > div:focus-within', rendered_page)
        self.assertIn("caret-color: var(--dm-accent) !important;", rendered_page)
        self.assertIn("--dm-chat-attachment-text: #202124;", rendered_page)
        self.assertIn("--dm-chat-icon: #276f66;", rendered_page)
        self.assertIn('[data-testid="stChatInput"] button svg', rendered_page)
        self.assertIn(
            '[data-testid="stChatInput"] [data-testid="stChatInputFile"] > div:first-child',
            rendered_page,
        )
        self.assertIn(".dm-save-note", rendered_page)

    def test_sidebar_uses_theme_from_query_params(self) -> None:
        fake_st = _FakeStreamlit()
        fake_st.query_params["theme"] = "dark"

        with patch.object(streamlit_sidebar, "st", fake_st), patch.object(
            streamlit_theme, "st", fake_st
        ):
            sidebar_inputs = streamlit_sidebar.render_sidebar()

        self.assertEqual(fake_st.session_state["documate_theme_mode"], "다크")
        self.assertEqual(sidebar_inputs.theme_mode, "다크")

    def test_quick_prompts_are_sampled_once_per_session(self) -> None:
        fake_st = _FakeStreamlit()
        sampled_prompts = [
            "추천 1",
            "추천 2",
            "추천 3",
            "추천 4",
        ]

        with patch.object(streamlit_intro, "st", fake_st), patch(
            "src.app.web.streamlit_intro.random.sample",
            side_effect=[
                sampled_prompts,
                ["다른 추천 1", "다른 추천 2", "다른 추천 3", "다른 추천 4"],
            ],
        ):
            streamlit_intro.render_intro({})
            streamlit_intro.render_intro({})

        self.assertEqual(fake_st.button_labels, sampled_prompts + sampled_prompts)


if __name__ == "__main__":
    unittest.main()
