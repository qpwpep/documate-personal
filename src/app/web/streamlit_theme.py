from __future__ import annotations

import streamlit as st


_THEME_OPTIONS = ["시스템", "라이트", "다크"]
_THEME_QUERY_MAP = {
    "system": "시스템",
    "light": "라이트",
    "dark": "다크",
    "시스템": "시스템",
    "라이트": "라이트",
    "다크": "다크",
}

_LIGHT_THEME_VARS = {
    "dm-bg": "#f7f5ef",
    "dm-panel": "#fffdfa",
    "dm-panel-soft": "#f1eee7",
    "dm-border": "#ddd7cb",
    "dm-text": "#202124",
    "dm-muted": "#6d675e",
    "dm-user": "#e8f0fe",
    "dm-accent": "#276f66",
    "dm-accent-soft": "#e4f2ef",
    "dm-accent-contrast": "#fffdfa",
    "dm-warm": "#f7e8d8",
    "dm-shadow": "0 18px 50px rgba(32, 33, 36, 0.08)",
    "dm-app-top": "#fbfaf7",
    "dm-app-glow": "rgba(39, 111, 102, 0.08)",
    "dm-sidebar-bg": "#f1eee7",
    "dm-divider": "rgba(221, 215, 203, 0.72)",
    "dm-strong-divider": "rgba(221, 215, 203, 0.86)",
    "dm-mark-bg": "#202124",
    "dm-mark-text": "#fffdfa",
    "dm-status-border": "rgba(39, 111, 102, 0.16)",
    "dm-button-hover-bg": "#ffffff",
    "dm-button-hover-border": "rgba(39, 111, 102, 0.42)",
    "dm-focus-ring": "rgba(39, 111, 102, 0.14)",
    "dm-user-border": "rgba(70, 116, 186, 0.16)",
    "dm-assistant-bg": (
        "linear-gradient(180deg, rgba(255, 253, 250, 0.76), "
        "rgba(247, 245, 239, 0.62))"
    ),
    "dm-assistant-border": "rgba(221, 215, 203, 0.52)",
    "dm-assistant-shadow": "0 10px 24px rgba(32, 33, 36, 0.032)",
    "dm-inline-code-bg": "rgba(39, 111, 102, 0.10)",
    "dm-inline-code-border": "rgba(39, 111, 102, 0.16)",
    "dm-inline-code-text": "#215f58",
    "dm-chat-input-bg": "#fffdfa",
    "dm-chat-input-gradient": (
        "linear-gradient(180deg, rgba(247, 245, 239, 0), "
        "rgba(247, 245, 239, 0.95) 22%)"
    ),
    "dm-upload-border": "rgba(39, 111, 102, 0.35)",
    "dm-chat-attachment-bg": "#f8f6f1",
    "dm-chat-attachment-text": "#202124",
    "dm-chat-attachment-muted": "#6d675e",
    "dm-chat-attachment-border": "rgba(39, 111, 102, 0.18)",
    "dm-chat-attachment-shadow": "0 6px 16px rgba(32, 33, 36, 0.08)",
    "dm-chat-icon": "#276f66",
    "dm-chat-icon-muted": "#6d675e",
}

_DARK_THEME_VARS = {
    "dm-bg": "#101214",
    "dm-panel": "#1b1c1e",
    "dm-panel-soft": "#151719",
    "dm-border": "#343230",
    "dm-text": "#f5f1e8",
    "dm-muted": "#b6afa4",
    "dm-user": "#193e39",
    "dm-accent": "#78d1c1",
    "dm-accent-soft": "#18322e",
    "dm-accent-contrast": "#071211",
    "dm-warm": "#2a211a",
    "dm-shadow": "0 18px 50px rgba(0, 0, 0, 0.38)",
    "dm-app-top": "#171613",
    "dm-app-glow": "rgba(120, 209, 193, 0.12)",
    "dm-sidebar-bg": "#151413",
    "dm-divider": "rgba(255, 255, 255, 0.10)",
    "dm-strong-divider": "rgba(255, 255, 255, 0.14)",
    "dm-mark-bg": "#f5f1e8",
    "dm-mark-text": "#111214",
    "dm-status-border": "rgba(120, 209, 193, 0.22)",
    "dm-button-hover-bg": "#222222",
    "dm-button-hover-border": "rgba(120, 209, 193, 0.50)",
    "dm-focus-ring": "rgba(120, 209, 193, 0.22)",
    "dm-user-border": "rgba(120, 209, 193, 0.18)",
    "dm-assistant-bg": (
        "linear-gradient(180deg, rgba(30, 32, 34, 0.62), "
        "rgba(22, 24, 25, 0.46))"
    ),
    "dm-assistant-border": "rgba(255, 255, 255, 0.08)",
    "dm-assistant-shadow": "0 12px 26px rgba(0, 0, 0, 0.11)",
    "dm-inline-code-bg": "rgba(120, 209, 193, 0.12)",
    "dm-inline-code-border": "rgba(120, 209, 193, 0.18)",
    "dm-inline-code-text": "#9fe6d8",
    "dm-chat-input-bg": "#1d1e20",
    "dm-chat-input-gradient": (
        "linear-gradient(180deg, rgba(16, 18, 20, 0), "
        "rgba(16, 18, 20, 0.96) 22%)"
    ),
    "dm-upload-border": "rgba(120, 209, 193, 0.35)",
    "dm-chat-attachment-bg": "#242827",
    "dm-chat-attachment-text": "#f5f1e8",
    "dm-chat-attachment-muted": "#b6afa4",
    "dm-chat-attachment-border": "rgba(120, 209, 193, 0.24)",
    "dm-chat-attachment-shadow": "0 6px 16px rgba(0, 0, 0, 0.18)",
    "dm-chat-icon": "#78d1c1",
    "dm-chat-icon-muted": "#b6afa4",
}



def render_theme_styles(theme_mode: str) -> None:
    if theme_mode == "라이트":
        st.markdown(_theme_vars_style(_LIGHT_THEME_VARS), unsafe_allow_html=True)
    elif theme_mode == "다크":
        st.markdown(_theme_vars_style(_DARK_THEME_VARS), unsafe_allow_html=True)



def _theme_vars_style(theme_vars: dict[str, str]) -> str:
    variables = "\n".join(f"    --{name}: {value};" for name, value in theme_vars.items())
    return f"<style>:root {{\n{variables}\n}}</style>"



def _sync_theme_from_query_params() -> None:
    raw_theme = st.query_params.get("theme")
    if isinstance(raw_theme, list):
        raw_theme = raw_theme[0] if raw_theme else None
    if raw_theme is None:
        return

    theme_mode = _THEME_QUERY_MAP.get(str(raw_theme).strip().lower())
    if theme_mode:
        st.session_state["documate_theme_mode"] = theme_mode

