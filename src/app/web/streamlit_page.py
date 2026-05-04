from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from html import escape

import streamlit as st

from src.infra.logging_utils import log_event


logger = logging.getLogger(__name__)


_QUICK_PROMPTS = [
    "pandas merge 사용법을 공식 문서 기준으로 설명해줘",
    "업로드한 노트북에서 pandas concat 예제를 찾아줘",
    "matplotlib pie 차트 옵션을 정리해줘",
    "방금 답변을 txt 파일로 저장해줘",
]

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
}


_APP_CSS = """
<style>
:root {
    --dm-bg: #f7f5ef;
    --dm-panel: #fffdfa;
    --dm-panel-soft: #f1eee7;
    --dm-border: #ddd7cb;
    --dm-text: #202124;
    --dm-muted: #6d675e;
    --dm-user: #e8f0fe;
    --dm-accent: #276f66;
    --dm-accent-soft: #e4f2ef;
    --dm-accent-contrast: #fffdfa;
    --dm-warm: #f7e8d8;
    --dm-shadow: 0 18px 50px rgba(32, 33, 36, 0.08);
    --dm-app-top: #fbfaf7;
    --dm-app-glow: rgba(39, 111, 102, 0.08);
    --dm-sidebar-bg: #f1eee7;
    --dm-divider: rgba(221, 215, 203, 0.72);
    --dm-strong-divider: rgba(221, 215, 203, 0.86);
    --dm-mark-bg: #202124;
    --dm-mark-text: #fffdfa;
    --dm-status-border: rgba(39, 111, 102, 0.16);
    --dm-button-hover-bg: #ffffff;
    --dm-button-hover-border: rgba(39, 111, 102, 0.42);
    --dm-focus-ring: rgba(39, 111, 102, 0.14);
    --dm-user-border: rgba(70, 116, 186, 0.16);
    --dm-assistant-bg: linear-gradient(180deg, rgba(255, 253, 250, 0.76), rgba(247, 245, 239, 0.62));
    --dm-assistant-border: rgba(221, 215, 203, 0.52);
    --dm-assistant-shadow: 0 10px 24px rgba(32, 33, 36, 0.032);
    --dm-inline-code-bg: rgba(39, 111, 102, 0.10);
    --dm-inline-code-border: rgba(39, 111, 102, 0.16);
    --dm-inline-code-text: #215f58;
    --dm-chat-input-bg: #fffdfa;
    --dm-chat-input-gradient: linear-gradient(180deg, rgba(247, 245, 239, 0), rgba(247, 245, 239, 0.95) 22%);
    --dm-upload-border: rgba(39, 111, 102, 0.35);
}

@media (prefers-color-scheme: dark) {
    :root {
        --dm-bg: #101214;
        --dm-panel: #1b1c1e;
        --dm-panel-soft: #151719;
        --dm-border: #343230;
        --dm-text: #f5f1e8;
        --dm-muted: #b6afa4;
        --dm-user: #193e39;
        --dm-accent: #78d1c1;
        --dm-accent-soft: #18322e;
        --dm-accent-contrast: #071211;
        --dm-warm: #2a211a;
        --dm-shadow: 0 18px 50px rgba(0, 0, 0, 0.38);
        --dm-app-top: #171613;
        --dm-app-glow: rgba(120, 209, 193, 0.12);
        --dm-sidebar-bg: #151413;
        --dm-divider: rgba(255, 255, 255, 0.10);
        --dm-strong-divider: rgba(255, 255, 255, 0.14);
        --dm-mark-bg: #f5f1e8;
        --dm-mark-text: #111214;
        --dm-status-border: rgba(120, 209, 193, 0.22);
        --dm-button-hover-bg: #222222;
        --dm-button-hover-border: rgba(120, 209, 193, 0.50);
        --dm-focus-ring: rgba(120, 209, 193, 0.22);
        --dm-user-border: rgba(120, 209, 193, 0.18);
        --dm-assistant-bg: linear-gradient(180deg, rgba(30, 32, 34, 0.62), rgba(22, 24, 25, 0.46));
        --dm-assistant-border: rgba(255, 255, 255, 0.08);
        --dm-assistant-shadow: 0 12px 26px rgba(0, 0, 0, 0.11);
        --dm-inline-code-bg: rgba(120, 209, 193, 0.12);
        --dm-inline-code-border: rgba(120, 209, 193, 0.18);
        --dm-inline-code-text: #9fe6d8;
        --dm-chat-input-bg: #1d1e20;
        --dm-chat-input-gradient: linear-gradient(180deg, rgba(16, 18, 20, 0), rgba(16, 18, 20, 0.96) 22%);
        --dm-upload-border: rgba(120, 209, 193, 0.35);
    }
}

#MainMenu,
footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.stApp {
    background:
        radial-gradient(circle at top left, var(--dm-app-glow), transparent 34rem),
        linear-gradient(180deg, var(--dm-app-top) 0%, var(--dm-bg) 100%);
    color: var(--dm-text);
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: var(--dm-sidebar-bg);
    border-right: 1px solid var(--dm-border);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 1.15rem 1rem 1.5rem;
}

button[data-testid="stBaseButton-headerNoPadding"],
button[data-testid="stExpandSidebarButton"] {
    background: var(--dm-accent-soft) !important;
    border: 1px solid var(--dm-status-border) !important;
    border-radius: 0.65rem !important;
    color: var(--dm-accent) !important;
    opacity: 1 !important;
}

button[data-testid="stBaseButton-headerNoPadding"] *,
button[data-testid="stExpandSidebarButton"] * {
    color: var(--dm-accent) !important;
    fill: currentColor !important;
    opacity: 1 !important;
    -webkit-text-fill-color: currentColor !important;
}

button[data-testid="stBaseButton-headerNoPadding"] [data-testid="stIconMaterial"],
button[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
    color: var(--dm-accent) !important;
    text-shadow: 0 0 14px var(--dm-focus-ring);
}

button[data-testid="stBaseButton-headerNoPadding"]:hover,
button[data-testid="stExpandSidebarButton"]:hover {
    background: var(--dm-panel) !important;
    border-color: var(--dm-button-hover-border) !important;
}

.block-container {
    max-width: 920px;
    padding: 2.2rem 1.5rem 8rem;
}

.dm-topbar {
    align-items: center;
    border-bottom: 1px solid var(--dm-divider);
    display: flex;
    gap: 0.75rem;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.9rem;
}

.dm-topbar-title {
    align-items: center;
    display: flex;
    gap: 0.72rem;
    min-width: 0;
}

.dm-mark {
    align-items: center;
    background: var(--dm-mark-bg);
    border-radius: 0.85rem;
    color: var(--dm-mark-text);
    display: inline-flex;
    font-weight: 800;
    height: 2.35rem;
    justify-content: center;
    letter-spacing: 0;
    width: 2.35rem;
}

.dm-title-copy {
    min-width: 0;
}

.dm-title-copy strong {
    color: var(--dm-text);
    display: block;
    font-size: 1rem;
    letter-spacing: 0;
    line-height: 1.2;
}

.dm-title-copy span,
.dm-topbar-status {
    color: var(--dm-muted);
    font-size: 0.84rem;
    letter-spacing: 0;
}

.dm-topbar-status {
    background: var(--dm-accent-soft);
    border: 1px solid var(--dm-status-border);
    border-radius: 999px;
    color: var(--dm-accent);
    padding: 0.42rem 0.72rem;
    white-space: nowrap;
}

.dm-sidebar-brand {
    border-bottom: 1px solid var(--dm-strong-divider);
    margin-bottom: 1.1rem;
    padding-bottom: 1rem;
}

.dm-sidebar-brand .dm-mark {
    margin-bottom: 0.7rem;
}

.dm-sidebar-brand h2 {
    color: var(--dm-text);
    font-size: 1.12rem;
    letter-spacing: 0;
    line-height: 1.3;
    margin: 0;
}

.dm-sidebar-brand p {
    color: var(--dm-muted);
    font-size: 0.9rem;
    line-height: 1.5;
    margin: 0.35rem 0 0;
}

.dm-sidebar-section {
    color: var(--dm-muted);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin: 1.05rem 0 0.45rem;
    text-transform: uppercase;
}

.dm-upload-note {
    background: var(--dm-panel);
    border: 1px solid var(--dm-border);
    border-radius: 0.75rem;
    color: var(--dm-muted);
    font-size: 0.86rem;
    line-height: 1.5;
    margin-top: 0.7rem;
    padding: 0.72rem 0.8rem;
}

.dm-intro {
    margin: 3.4rem auto 1.3rem;
    max-width: 780px;
    text-align: center;
}

.dm-intro-kicker {
    color: var(--dm-accent);
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 0.7rem;
    text-transform: uppercase;
}

.dm-intro h1 {
    color: var(--dm-text);
    font-size: 3rem;
    letter-spacing: 0;
    line-height: 1.12;
    margin: 0;
}

.dm-intro p {
    color: var(--dm-muted);
    font-size: 1.06rem;
    line-height: 1.7;
    margin: 1rem auto 1.35rem;
    max-width: 610px;
}

.dm-docs-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin: 0.6rem auto 1.15rem;
}

.dm-docs-row span {
    background: var(--dm-panel);
    border: 1px solid var(--dm-border);
    border-radius: 999px;
    color: var(--dm-muted);
    font-size: 0.82rem;
    padding: 0.34rem 0.7rem;
}

div.stButton > button {
    background: var(--dm-panel);
    border: 1px solid var(--dm-border);
    border-radius: 0.9rem;
    box-shadow: none;
    color: var(--dm-text);
    min-height: 3.25rem;
    padding: 0.72rem 0.9rem;
    text-align: left;
    transition: border-color 120ms ease, background 120ms ease, transform 120ms ease;
    white-space: normal;
}

div.stButton > button:hover {
    background: var(--dm-button-hover-bg);
    border-color: var(--dm-button-hover-border);
    color: var(--dm-text);
    transform: translateY(-1px);
}

div.stButton > button:focus {
    border-color: var(--dm-accent);
    box-shadow: 0 0 0 0.14rem var(--dm-focus-ring);
}

[data-testid="stChatMessage"] {
    background: transparent;
    border: 0;
    margin: 0.45rem 0;
    padding: 0.25rem 0;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: var(--dm-text);
    line-height: 1.72;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
    background: var(--dm-mark-bg);
    color: var(--dm-mark-text);
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    margin-left: auto;
    max-width: 78%;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
    background: var(--dm-user);
    border: 1px solid var(--dm-user-border);
    border-radius: 1.1rem;
    padding: 0.82rem 1rem;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] {
    backdrop-filter: blur(10px);
    background: var(--dm-assistant-bg);
    border: 1px solid var(--dm-assistant-border);
    border-radius: 1.1rem;
    box-shadow: var(--dm-assistant-shadow);
    padding: 0.9rem 1rem;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
    background: var(--dm-inline-code-bg) !important;
    border: 1px solid var(--dm-inline-code-border);
    border-radius: 0.34rem;
    color: var(--dm-inline-code-text) !important;
    font-size: 0.88em;
    padding: 0.08rem 0.28rem;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
    background: var(--dm-inline-code-bg) !important;
    border: 1px solid var(--dm-inline-code-border);
    border-radius: 0.7rem;
    color: var(--dm-inline-code-text) !important;
    padding: 0.72rem 0.86rem;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre code {
    background: transparent !important;
    border: 0;
    border-radius: 0;
    color: var(--dm-inline-code-text) !important;
    display: block;
    padding: 0;
    white-space: pre-wrap;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stAlert"] {
    backdrop-filter: blur(10px);
    background: var(--dm-assistant-bg) !important;
    border: 1px solid var(--dm-assistant-border) !important;
    border-radius: 1.1rem !important;
    box-shadow: var(--dm-assistant-shadow);
    color: var(--dm-text) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stAlert"] * {
    color: var(--dm-text) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stAlert"] code {
    background: var(--dm-inline-code-bg) !important;
    border: 1px solid var(--dm-inline-code-border);
    color: var(--dm-inline-code-text) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"]:has(.dm-download-button) {
    background: transparent;
    border: 0;
    box-shadow: none;
    padding: 0.28rem 0 0;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"]:has(.dm-save-note) {
    background: transparent;
    border: 0;
    box-shadow: none;
    padding: 0.28rem 0 0;
}

[data-testid="stExpander"] {
    background: var(--dm-panel);
    border: 1px solid var(--dm-border);
    border-radius: 0.9rem;
    box-shadow: none;
}

[data-testid="stChatInput"] {
    background: transparent !important;
    overflow: visible !important;
    padding-bottom: 1.1rem;
}

[data-testid="stBottom"] > div {
    background: var(--dm-chat-input-gradient) !important;
    overflow: visible !important;
}

[data-testid="stChatInput"] > div {
    background-color: var(--dm-chat-input-bg) !important;
    border: 1px solid var(--dm-border);
    border-radius: 1rem !important;
    box-shadow: var(--dm-shadow);
    overflow: visible !important;
    padding: 0.45rem !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--dm-accent) !important;
    box-shadow: var(--dm-shadow), 0 0 0 0.12rem var(--dm-focus-ring) !important;
    outline: none !important;
}

[data-testid="stChatInput"] > div > div {
    background-color: transparent !important;
    border-radius: 0.9rem !important;
    overflow: visible !important;
}

[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
    background-color: transparent !important;
    border-radius: 0.9rem !important;
    overflow: visible !important;
}

[data-testid="stChatInput"] [data-baseweb="textarea"]:focus,
[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within,
[data-testid="stChatInput"] [data-baseweb="base-input"]:focus,
[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
    border-color: transparent !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] div {
    color: var(--dm-text) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: 0 !important;
    border-radius: 0.9rem !important;
    box-shadow: none !important;
    color: var(--dm-text) !important;
    min-height: 3.2rem;
}

[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {
    border-color: transparent !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: var(--dm-muted);
    opacity: 0.8;
}

[data-testid="stChatInput"] button {
    background: var(--dm-accent-soft) !important;
    border-radius: 0.72rem !important;
    color: var(--dm-accent) !important;
}

[data-testid="stFileUploader"] {
    background: var(--dm-panel);
    border: 1px dashed var(--dm-upload-border);
    border-radius: 0.9rem;
    padding: 0.4rem;
}

[data-testid="stTextInput"] input {
    background: var(--dm-chat-input-bg);
    border-color: var(--dm-border);
    border-radius: 0.7rem;
    color: var(--dm-text);
}

[data-testid="stRadio"] label,
[data-testid="stTextInput"] label {
    color: var(--dm-text);
}

[data-testid="stRadio"] [role="radiogroup"] {
    background: var(--dm-panel);
    border: 1px solid var(--dm-border);
    border-radius: 0.85rem;
    gap: 0.25rem;
    padding: 0.24rem;
}

[data-testid="stRadio"] [role="radio"] {
    color: var(--dm-muted);
}

[data-testid="stRadio"] [role="radio"] * {
    color: var(--dm-muted) !important;
}

[data-testid="stRadio"] [aria-checked="true"] {
    background: var(--dm-accent-soft);
    border-radius: 0.64rem;
    color: var(--dm-accent);
}

[data-testid="stRadio"] [aria-checked="true"] * {
    color: var(--dm-accent) !important;
}

.dm-download-button {
    background-color: var(--dm-accent);
    border: none;
    border-radius: 0.95rem;
    color: var(--dm-accent-contrast);
    cursor: pointer;
    font-size: 0.95rem;
    padding: 0.75rem 1rem;
    width: 100%;
}

.dm-save-note {
    backdrop-filter: blur(10px);
    background: var(--dm-assistant-bg);
    border: 1px solid var(--dm-assistant-border);
    border-radius: 1rem;
    box-shadow: var(--dm-assistant-shadow);
    color: var(--dm-text);
    margin: 0 0 0.75rem;
    padding: 0.9rem 1rem;
}

.dm-save-note code {
    background: var(--dm-inline-code-bg) !important;
    border: 1px solid var(--dm-inline-code-border);
    border-radius: 0.34rem;
    color: var(--dm-inline-code-text) !important;
    padding: 0.08rem 0.28rem;
}

.stAlert {
    border-radius: 0.85rem;
}

@media (max-width: 760px) {
    .block-container {
        padding: 1.2rem 1rem 7.5rem;
    }

    .dm-topbar {
        align-items: flex-start;
        flex-direction: column;
    }

    .dm-topbar-status {
        white-space: normal;
    }

    .dm-intro {
        margin-top: 2.4rem;
        text-align: left;
    }

    .dm-intro h1 {
        font-size: 2.25rem;
    }

    .dm-docs-row {
        justify-content: flex-start;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        max-width: 100%;
    }
}
</style>
"""


@dataclass
class SidebarInputs:
    slack_user_id: str
    slack_email: str
    slack_channel_id: str
    theme_mode: str


def configure_page() -> None:
    st.set_page_config(page_title="DocuMate", layout="wide")
    st.markdown(_APP_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="dm-topbar">
            <div class="dm-topbar-title">
                <span class="dm-mark">DM</span>
                <div class="dm-title-copy">
                    <strong>DocuMate</strong>
                    <span>공식 문서와 업로드한 코드로 답하는 학습 채팅</span>
                </div>
            </div>
            <div class="dm-topbar-status">FastAPI + Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    _sync_theme_from_query_params()

    with st.sidebar:
        st.markdown(
            """
            <div class="dm-sidebar-brand">
                <span class="dm-mark">DM</span>
                <h2>DocuMate</h2>
                <p>문서를 근거로 답하고, 필요한 결과를 파일이나 Slack으로 이어서 보냅니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="dm-sidebar-section">현재 파일</div>', unsafe_allow_html=True)
        uploaded_label = escape(current_file_name or "아직 업로드된 파일이 없습니다.")
        st.markdown(
            f'<div class="dm-upload-note">현재 파일: <strong>{uploaded_label}</strong></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="dm-sidebar-section">화면 모드</div>', unsafe_allow_html=True)
        theme_mode = st.radio(
            "테마",
            options=_THEME_OPTIONS,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="documate_theme_mode",
        )

        st.markdown('<div class="dm-sidebar-section">Slack 전송</div>', unsafe_allow_html=True)
        slack_user_id = st.text_input("User ID", value="", placeholder="Uxxxxx")
        slack_email = st.text_input("Email", value="", placeholder="name@example.com")
        slack_channel_id = st.text_input("Channel ID", value="", placeholder="C/G/Dxxxxx")

    return SidebarInputs(
        slack_user_id=slack_user_id,
        slack_email=slack_email,
        slack_channel_id=slack_channel_id,
        theme_mode=theme_mode,
    )


def render_theme_styles(theme_mode: str) -> None:
    if theme_mode == "라이트":
        st.markdown(_theme_vars_style(_LIGHT_THEME_VARS), unsafe_allow_html=True)
    elif theme_mode == "다크":
        st.markdown(_theme_vars_style(_DARK_THEME_VARS), unsafe_allow_html=True)


def render_intro(default_docs: dict[str, str]) -> str | None:
    docs_badges = "".join(f"<span>{escape(key)}</span>" for key in list(default_docs.keys()))
    st.markdown(
        f"""
        <section class="dm-intro">
            <div class="dm-intro-kicker">DocuMate</div>
            <h1>무엇을 도와드릴까요?</h1>
            <p>공식 문서, 로컬 인덱스, 업로드한 코드 파일을 함께 확인해 근거가 남는 답변을 만듭니다.</p>
            <div class="dm-docs-row">{docs_badges}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    selected_prompt: str | None = None
    for row_start in range(0, len(_QUICK_PROMPTS), 2):
        columns = st.columns(2)
        for offset, prompt in enumerate(_QUICK_PROMPTS[row_start : row_start + 2]):
            index = row_start + offset
            with columns[offset]:
                if st.button(prompt, key=f"quick_prompt_{index}", use_container_width=True):
                    selected_prompt = prompt
    return selected_prompt


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
