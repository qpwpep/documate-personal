from __future__ import annotations


THEME_OPTIONS = ("시스템", "라이트", "다크")
THEME_STATE_KEY = "documate_theme_mode"

_THEME_TOKENS = {
    "dm-bg": {
        "라이트": "#f7f5ef",
        "다크": "#101214",
    },
    "dm-panel": {
        "라이트": "#fffdfa",
        "다크": "#1b1c1e",
    },
    "dm-border": {
        "라이트": "#ddd7cb",
        "다크": "#343230",
    },
    "dm-text": {
        "라이트": "#202124",
        "다크": "#f5f1e8",
    },
    "dm-muted": {
        "라이트": "#6d675e",
        "다크": "#b6afa4",
    },
    "dm-user": {
        "라이트": "#e8f0fe",
        "다크": "#193e39",
    },
    "dm-accent": {
        "라이트": "#276f66",
        "다크": "#78d1c1",
    },
    "dm-accent-soft": {
        "라이트": "#e4f2ef",
        "다크": "#18322e",
    },
    "dm-shadow": {
        "라이트": "0 18px 50px rgba(32, 33, 36, 0.08)",
        "다크": "0 18px 50px rgba(0, 0, 0, 0.38)",
    },
    "dm-app-top": {
        "라이트": "#fbfaf7",
        "다크": "#171613",
    },
    "dm-app-glow": {
        "라이트": "rgba(39, 111, 102, 0.08)",
        "다크": "rgba(120, 209, 193, 0.12)",
    },
    "dm-sidebar-bg": {
        "라이트": "#f1eee7",
        "다크": "#151413",
    },
    "dm-divider": {
        "라이트": "rgba(221, 215, 203, 0.72)",
        "다크": "rgba(255, 255, 255, 0.10)",
    },
    "dm-strong-divider": {
        "라이트": "rgba(221, 215, 203, 0.86)",
        "다크": "rgba(255, 255, 255, 0.14)",
    },
    "dm-mark-bg": {
        "라이트": "#202124",
        "다크": "#f5f1e8",
    },
    "dm-mark-text": {
        "라이트": "#fffdfa",
        "다크": "#111214",
    },
    "dm-status-border": {
        "라이트": "rgba(39, 111, 102, 0.16)",
        "다크": "rgba(120, 209, 193, 0.22)",
    },
    "dm-button-hover-bg": {
        "라이트": "#ffffff",
        "다크": "#222222",
    },
    "dm-button-hover-border": {
        "라이트": "rgba(39, 111, 102, 0.42)",
        "다크": "rgba(120, 209, 193, 0.50)",
    },
    "dm-focus-ring": {
        "라이트": "rgba(39, 111, 102, 0.14)",
        "다크": "rgba(120, 209, 193, 0.22)",
    },
    "dm-user-border": {
        "라이트": "rgba(70, 116, 186, 0.16)",
        "다크": "rgba(120, 209, 193, 0.18)",
    },
    "dm-assistant-bg": {
        "라이트": "linear-gradient(180deg, rgba(255, 253, 250, 0.76), rgba(247, 245, 239, 0.62))",
        "다크": "linear-gradient(180deg, rgba(30, 32, 34, 0.62), rgba(22, 24, 25, 0.46))",
    },
    "dm-assistant-border": {
        "라이트": "rgba(221, 215, 203, 0.52)",
        "다크": "rgba(255, 255, 255, 0.08)",
    },
    "dm-assistant-shadow": {
        "라이트": "0 10px 24px rgba(32, 33, 36, 0.032)",
        "다크": "0 12px 26px rgba(0, 0, 0, 0.11)",
    },
    "dm-inline-code-bg": {
        "라이트": "rgba(39, 111, 102, 0.10)",
        "다크": "rgba(120, 209, 193, 0.12)",
    },
    "dm-inline-code-border": {
        "라이트": "rgba(39, 111, 102, 0.16)",
        "다크": "rgba(120, 209, 193, 0.18)",
    },
    "dm-inline-code-text": {
        "라이트": "#215f58",
        "다크": "#9fe6d8",
    },
    "dm-chat-input-bg": {
        "라이트": "#fffdfa",
        "다크": "#1d1e20",
    },
    "dm-chat-input-gradient": {
        "라이트": "linear-gradient(180deg, rgba(247, 245, 239, 0), rgba(247, 245, 239, 0.95) 22%)",
        "다크": "linear-gradient(180deg, rgba(16, 18, 20, 0), rgba(16, 18, 20, 0.96) 22%)",
    },
    "dm-upload-border": {
        "라이트": "rgba(39, 111, 102, 0.35)",
        "다크": "rgba(120, 209, 193, 0.35)",
    },
    "dm-chat-attachment-bg": {
        "라이트": "#f8f6f1",
        "다크": "#242827",
    },
    "dm-chat-attachment-text": {
        "라이트": "#202124",
        "다크": "#f5f1e8",
    },
    "dm-chat-attachment-muted": {
        "라이트": "#6d675e",
        "다크": "#b6afa4",
    },
    "dm-chat-attachment-border": {
        "라이트": "rgba(39, 111, 102, 0.18)",
        "다크": "rgba(120, 209, 193, 0.24)",
    },
    "dm-chat-attachment-shadow": {
        "라이트": "0 6px 16px rgba(32, 33, 36, 0.08)",
        "다크": "0 6px 16px rgba(0, 0, 0, 0.18)",
    },
    "dm-chat-icon": {
        "라이트": "#276f66",
        "다크": "#78d1c1",
    },
    "dm-chat-icon-muted": {
        "라이트": "#6d675e",
        "다크": "#b6afa4",
    },
}


def build_theme_css(theme_mode: str) -> str:
    """Build the complete token rules for the selected preference."""
    if theme_mode not in THEME_OPTIONS:
        raise ValueError(f"Unsupported theme mode: {theme_mode!r}")
    if theme_mode != "시스템":
        return _theme_vars_css(theme_mode)

    return (
        f"{_theme_vars_css('라이트')}\n\n"
        "@media (prefers-color-scheme: dark) {\n"
        f"{_theme_vars_css('다크')}\n"
        "}"
    )


def _theme_vars_css(theme_mode: str) -> str:
    variables = "\n".join(
        f"    --{name}: {values[theme_mode]};"
        for name, values in _THEME_TOKENS.items()
    )
    return f":root {{\n{variables}\n}}"
