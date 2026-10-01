from __future__ import annotations

import re
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from src.app.web import streamlit_intro, streamlit_sidebar, streamlit_styles, streamlit_theme


class _FakeStreamlit:
    def __init__(self) -> None:
        self.markdowns: list[tuple[str, bool]] = []
        self.session_state: dict[str, object] = {}
        self.button_labels: list[str] = []
        self.input_values: dict[str, str] = {}
        self.errors: list[str] = []
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
        return self.input_values.get(label, value)

    def error(self, message: str) -> None:
        self.errors.append(message)


class StreamlitPageTest(unittest.TestCase):
    def test_sidebar_selects_one_typed_slack_recipient(self) -> None:
        fake_st = _FakeStreamlit()
        fake_st.session_state["documate_slack_recipient_kind"] = "이메일"
        fake_st.input_values["Slack 수신자"] = "selected@example.com"
        with patch.object(streamlit_sidebar, "st", fake_st):
            selected = streamlit_sidebar.render_sidebar()
        self.assertEqual(selected.slack_recipient.model_dump(), {"kind": "email", "value": "selected@example.com"})
        self.assertIsNone(selected.slack_recipient_error)

    def test_sidebar_blank_explicit_recipient_is_an_error(self) -> None:
        fake_st = _FakeStreamlit()
        fake_st.session_state["documate_slack_recipient_kind"] = "사용자"
        with patch.object(streamlit_sidebar, "st", fake_st):
            selected = streamlit_sidebar.render_sidebar()
        self.assertIsNone(selected.slack_recipient)
        self.assertIsNotNone(selected.slack_recipient_error)
        self.assertEqual(fake_st.errors, [selected.slack_recipient_error])

    def test_sidebar_unspecified_recipient_has_no_input_error(self) -> None:
        fake_st = _FakeStreamlit()
        with patch.object(streamlit_sidebar, "st", fake_st):
            selected = streamlit_sidebar.render_sidebar()
        self.assertIsNone(selected.slack_recipient)
        self.assertIsNone(selected.slack_recipient_error)

    def test_configure_page_renders_selected_theme_with_component_styles(self) -> None:
        for theme_mode in ("시스템", "라이트", "다크"):
            with self.subTest(theme_mode=theme_mode):
                fake_st = _FakeStreamlit()
                with patch.object(streamlit_styles, "st", fake_st):
                    streamlit_styles.configure_page(theme_mode)

                styles = [entry for entry in fake_st.markdowns if "<style>" in entry[0]]
                self.assertEqual(len(styles), 1)
                rendered_page, unsafe = styles[0]
                self.assertTrue(unsafe)
                self.assertEqual(rendered_page.count("<style>"), 1)
                self.assertEqual(rendered_page.count("</style>"), 1)
                theme_css = streamlit_theme.build_theme_css(theme_mode)
                self.assertIn(theme_css, rendered_page)
                self.assertEqual("prefers-color-scheme" in rendered_page, theme_mode == "시스템")
                component_css = rendered_page.replace(theme_css, "", 1)
                self.assertNotRegex(component_css, r"--dm-[\w-]+\s*:")
                defined_tokens = set(re.findall(r"(--dm-[\w-]+)\s*:", theme_css))
                used_tokens = set(re.findall(r"var\((--dm-[\w-]+)\)", component_css))
                self.assertLessEqual(used_tokens, defined_tokens)
                self.assertIn('[data-testid="stChatInput"] > div:focus-within', component_css)
                self.assertIn("caret-color: var(--dm-accent) !important;", component_css)
                self.assertIn('[data-testid="stChatInput"] button svg', component_css)
                self.assertIn(
                    '[data-testid="stChatInput"] [data-testid="stChatInputFile"] > div:first-child',
                    component_css,
                )

    def test_native_code_and_popover_surfaces_use_theme_tokens(self) -> None:
        fake_st = _FakeStreamlit()
        with patch.object(streamlit_styles, "st", fake_st):
            streamlit_styles.configure_page("시스템")
        css, = [body for body, _ in fake_st.markdowns if "<style>" in body]
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)

        for selector, expected_background, expected_color in (
            ('[data-testid="stCode"] pre', "var(--dm-inline-code-bg)", "var(--dm-inline-code-text)"),
            ('[data-testid="stCode"] pre code', "transparent", "var(--dm-inline-code-text)"),
            ('[data-testid="stPopoverBody"] > div', "var(--dm-panel)", "var(--dm-text)"),
            ('[data-testid="stCode"] .react-syntax-highlighter-line-number', None, "var(--dm-muted)"),
        ):
            with self.subTest(selector=selector):
                declarations = {}
                for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
                    if selector in [item.strip() for item in selectors.split(",")]:
                        declarations.update(re.findall(r"([\w-]+)\s*:\s*([^;]+);", body))
                self.assertTrue(declarations, f"Missing native surface rule: {selector}")
                if expected_background is not None:
                    background = declarations.get("background", declarations.get("background-color", ""))
                    self.assertEqual(background.removesuffix("!important").strip(), expected_background)
                self.assertEqual(
                    declarations.get("color", "").removesuffix("!important").strip(),
                    expected_color,
                )

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
