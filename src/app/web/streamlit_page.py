from __future__ import annotations

import logging
import random
import sys
from functools import lru_cache

import streamlit as st

from src.app.web import streamlit_intro as _intro
from src.app.web import streamlit_sidebar as _sidebar
from src.app.web import streamlit_styles as _styles
from src.app.web import streamlit_theme as _theme
from src.infra.logging_utils import log_event


logger = logging.getLogger(__name__)

_APP_CSS = _styles._APP_CSS
_DARK_THEME_VARS = _theme._DARK_THEME_VARS
_LIGHT_THEME_VARS = _theme._LIGHT_THEME_VARS
_QUICK_PROMPT_COUNT = _intro._QUICK_PROMPT_COUNT
_QUICK_PROMPTS = _intro._QUICK_PROMPTS
_THEME_OPTIONS = _theme._THEME_OPTIONS
_THEME_QUERY_MAP = _theme._THEME_QUERY_MAP
SidebarInputs = _sidebar.SidebarInputs


def configure_page() -> None:
    _styles.st = st
    _styles.configure_page()


@lru_cache(maxsize=1)
def warn_if_utf8_mode_disabled_once() -> None:
    if sys.flags.utf8_mode == 1:
        return

    log_event(
        logger,
        logging.WARNING,
        "utf8_mode_disabled",
        suggested_command=(
            "uv run python -X utf8 -m streamlit run src/app/web/streamlit_app.py --server.port 8501"
        ),
    )


def render_sidebar(current_file_name: str | None = None) -> SidebarInputs:
    _sidebar.st = st
    _theme.st = st
    return _sidebar.render_sidebar(current_file_name)


def render_theme_styles(theme_mode: str) -> None:
    _theme.st = st
    _theme.render_theme_styles(theme_mode)


def render_intro(default_docs: dict[str, str]) -> str | None:
    _intro.st = st
    _intro.random = random
    return _intro.render_intro(default_docs)


def _get_quick_prompts_for_session() -> list[str]:
    _intro.st = st
    _intro.random = random
    return _intro._get_quick_prompts_for_session()


def _theme_vars_style(theme_vars: dict[str, str]) -> str:
    return _theme._theme_vars_style(theme_vars)


def _sync_theme_from_query_params() -> None:
    _theme.st = st
    _theme._sync_theme_from_query_params()
