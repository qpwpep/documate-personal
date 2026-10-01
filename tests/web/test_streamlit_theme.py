"""Theme CSS and selection through the real Streamlit application."""

import re

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.app.web import streamlit_app, streamlit_state, streamlit_theme
from src.core.uploads import UploadManifest


@pytest.fixture
def theme_app(monkeypatch, tmp_path):
    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path)

    def send(_session, method, url, **kwargs):
        assert method == "get"
        response = requests.Response()
        response.status_code = 200
        response._content = UploadManifest(epoch="theme-test", revision=0, files=[]).model_dump_json().encode()
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", send)
    return AppTest.from_file(streamlit_app.__file__)


def _root_declarations(css):
    return [
        dict(re.findall(r"([\w-]+)\s*:\s*([^;]+);", block))
        for block in re.findall(r":root\s*\{([^}]+)\}", css)
    ]


def _assert_selected_theme(app, mode):
    assert not app.exception
    assert app.radio(key=streamlit_theme.THEME_STATE_KEY).value == mode
    assert app.session_state[streamlit_theme.THEME_STATE_KEY] == mode
    styles = [item.value for item in app.markdown if "--dm-bg:" in item.value]
    assert len(styles) == 1
    assert streamlit_theme.build_theme_css(mode) in styles[0]
    assert styles[0].count("--dm-bg:") == (2 if mode == "시스템" else 1)
    assert ("prefers-color-scheme" in styles[0]) == (mode == "시스템")


@pytest.mark.parametrize(
    ("mode", "background", "text", "chat_input"),
    [
        ("라이트", "#f7f5ef", "#202124", "#fffdfa"),
        ("다크", "#101214", "#f5f1e8", "#1d1e20"),
    ],
)
def test_explicit_css_contains_only_the_selected_palette(mode, background, text, chat_input):
    css = streamlit_theme.build_theme_css(mode)
    assert "<style" not in css
    assert "prefers-color-scheme" not in css
    palettes = _root_declarations(css)
    assert len(palettes) == 1
    assert palettes[0]["--dm-bg"] == background
    assert palettes[0]["--dm-text"] == text
    assert palettes[0]["--dm-chat-input-bg"] == chat_input


def test_system_css_uses_the_same_complete_palettes_as_explicit_modes():
    light, = _root_declarations(streamlit_theme.build_theme_css("라이트"))
    dark, = _root_declarations(streamlit_theme.build_theme_css("다크"))
    system_css = streamlit_theme.build_theme_css("시스템")
    assert light.keys() == dark.keys()
    assert _root_declarations(system_css) == [light, dark]
    assert "@media (prefers-color-scheme: dark)" in system_css
    base_css, dark_media_css = system_css.split("@media (prefers-color-scheme: dark)")
    assert _root_declarations(base_css) == [light]
    assert _root_declarations(dark_media_css) == [dark]


def test_url_theme_only_seeds_initial_selection(theme_app):
    theme_app.query_params["theme"] = "dark"
    theme_app.run()
    assert not theme_app.exception
    assert theme_app.radio(key="documate_theme_mode").value == "다크"

    theme_app.radio(key="documate_theme_mode").set_value("라이트").run()

    assert not theme_app.exception
    assert theme_app.radio(key="documate_theme_mode").value == "라이트"
    assert theme_app.session_state["documate_theme_mode"] == "라이트"
    _assert_selected_theme(theme_app, "라이트")
    theme_app.run()
    _assert_selected_theme(theme_app, "라이트")

    theme_app.query_params["theme"] = "system"
    theme_app.run()
    _assert_selected_theme(theme_app, "라이트")


@pytest.mark.parametrize(
    ("query_value", "expected"),
    [
        (None, "시스템"),
        ("", "시스템"),
        ("unknown", "시스템"),
        ("system", "시스템"),
        ("light", "라이트"),
        ("dark", "다크"),
        ("시스템", "시스템"),
        ("라이트", "라이트"),
        ("다크", "다크"),
        (" LIGHT ", "라이트"),
    ],
)
def test_initial_theme_accepts_query_aliases_and_defaults_to_system(theme_app, query_value, expected):
    if query_value is not None:
        theme_app.query_params["theme"] = query_value
    theme_app.run()
    _assert_selected_theme(theme_app, expected)


def test_existing_session_selection_takes_precedence_over_url(theme_app):
    theme_app.query_params["theme"] = "dark"
    theme_app.session_state[streamlit_theme.THEME_STATE_KEY] = "라이트"
    theme_app.run()
    _assert_selected_theme(theme_app, "라이트")


def test_theme_switches_on_the_current_rerun_and_survives_unrelated_widgets(theme_app):
    theme_app.run()
    _assert_selected_theme(theme_app, "시스템")
    for mode in ("라이트", "다크", "시스템", "다크", "라이트", "시스템"):
        theme_app.radio(key=streamlit_theme.THEME_STATE_KEY).set_value(mode).run()
        _assert_selected_theme(theme_app, mode)
        theme_app.run()
        _assert_selected_theme(theme_app, mode)

    theme_app.radio(key=streamlit_theme.THEME_STATE_KEY).set_value("다크").run()
    theme_app.radio(key="documate_slack_recipient_kind").set_value("채널").run()
    _assert_selected_theme(theme_app, "다크")
    theme_app.text_input(key="documate_slack_recipient_channel").set_value("C12345678").run()
    _assert_selected_theme(theme_app, "다크")


def test_new_chat_preserves_selected_theme(theme_app):
    theme_app.query_params["theme"] = "dark"
    theme_app.run()
    theme_app.radio(key=streamlit_theme.THEME_STATE_KEY).set_value("라이트").run()
    _assert_selected_theme(theme_app, "라이트")
    previous_session = theme_app.session_state["session_id"]

    theme_app.button(key="documate_new_chat").click().run()

    _assert_selected_theme(theme_app, "라이트")
    assert theme_app.session_state["session_id"] != previous_session
