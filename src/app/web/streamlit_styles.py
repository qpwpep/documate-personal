from __future__ import annotations

import streamlit as st


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
    --dm-chat-attachment-bg: #f8f6f1;
    --dm-chat-attachment-text: #202124;
    --dm-chat-attachment-muted: #6d675e;
    --dm-chat-attachment-border: rgba(39, 111, 102, 0.18);
    --dm-chat-attachment-shadow: 0 6px 16px rgba(32, 33, 36, 0.08);
    --dm-chat-icon: #276f66;
    --dm-chat-icon-muted: #6d675e;
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
        --dm-chat-attachment-bg: #242827;
        --dm-chat-attachment-text: #f5f1e8;
        --dm-chat-attachment-muted: #b6afa4;
        --dm-chat-attachment-border: rgba(120, 209, 193, 0.24);
        --dm-chat-attachment-shadow: 0 6px 16px rgba(0, 0, 0, 0.18);
        --dm-chat-icon: #78d1c1;
        --dm-chat-icon-muted: #b6afa4;
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
    margin-bottom: 0.9rem;
    padding-bottom: 0.95rem;
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
    line-height: 1.2;
    margin: 0.95rem 0 0.42rem;
    text-transform: uppercase;
}

.dm-upload-note {
    background: var(--dm-panel);
    border: 1px solid var(--dm-border);
    border-radius: 0.75rem;
    color: var(--dm-muted);
    font-size: 0.86rem;
    line-height: 1.5;
    margin: 0 0 1rem;
    padding: 0.72rem 0.8rem;
}

[data-testid="stSidebar"] div.stButton > button {
    min-height: 3rem;
}

[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(div.stButton),
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]) {
    margin-bottom: 0;
}

[data-testid="stSidebar"] [data-testid="stRadio"] {
    margin: 0;
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

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] > :first-child {
    margin-top: 0;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] > :last-child {
    margin-bottom: 0;
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
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 3.35rem;
    padding: 0.82rem 1rem;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] {
    backdrop-filter: blur(10px);
    background: var(--dm-assistant-bg);
    border: 1px solid var(--dm-assistant-border);
    border-radius: 1.1rem;
    box-shadow: var(--dm-assistant-shadow);
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 3.35rem;
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
    background: var(--dm-assistant-bg);
    border: 1px solid var(--dm-assistant-border);
    border-radius: 0.9rem;
    box-shadow: var(--dm-assistant-shadow);
    overflow: hidden;
}

[data-testid="stExpander"] details {
    background: transparent;
}

[data-testid="stExpander"] summary {
    background: transparent !important;
    color: var(--dm-text) !important;
    min-height: 3.1rem;
    padding: 0.68rem 0.9rem !important;
}

[data-testid="stExpander"] summary:hover {
    background: var(--dm-focus-ring) !important;
}

[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg {
    color: var(--dm-text) !important;
    fill: currentColor !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--dm-text) !important;
}

[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    display: block;
    min-height: 0;
    padding: 0 !important;
}

[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] p {
    margin: 0;
}

[data-testid="stExpanderDetails"] {
    background: transparent !important;
    border-top: 1px solid var(--dm-divider);
    color: var(--dm-text) !important;
}

[data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] {
    background: var(--dm-assistant-bg);
    border: 1px solid var(--dm-assistant-border);
    border-radius: 0.9rem;
    color: var(--dm-text);
    padding: 0.85rem 1rem;
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
    align-items: center !important;
    background-color: transparent !important;
    border-radius: 0.9rem !important;
    display: flex !important;
    min-height: 3.2rem;
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
    caret-color: var(--dm-accent) !important;
    color: var(--dm-text) !important;
    line-height: 1.45 !important;
    min-height: 3.2rem;
    padding: 0.84rem 0.75rem !important;
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
    color: var(--dm-chat-icon) !important;
}

[data-testid="stChatInput"] button *,
[data-testid="stChatInput"] button svg {
    color: var(--dm-chat-icon) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--dm-chat-icon) !important;
}

[data-testid="stChatInput"] button:disabled,
[data-testid="stChatInput"] button:disabled *,
[data-testid="stChatInput"] button[disabled],
[data-testid="stChatInput"] button[disabled] * {
    color: var(--dm-chat-icon-muted) !important;
    opacity: 0.68 !important;
    -webkit-text-fill-color: var(--dm-chat-icon-muted) !important;
}

[data-testid="stChatInput"] [data-testid="stUploadedFile"],
[data-testid="stChatInput"] [data-testid="stFileUploaderFile"],
[data-testid="stChatInput"] [data-testid="stChatInputFile"] {
    background: var(--dm-chat-attachment-bg) !important;
    border: 1px solid var(--dm-chat-attachment-border) !important;
    border-radius: 0.6rem !important;
    box-shadow: var(--dm-chat-attachment-shadow) !important;
    color: var(--dm-chat-attachment-text) !important;
}

[data-testid="stChatInput"] [data-testid="stUploadedFile"] *,
[data-testid="stChatInput"] [data-testid="stFileUploaderFile"] *,
[data-testid="stChatInput"] [data-testid="stChatInputFile"] * {
    color: var(--dm-chat-attachment-text) !important;
    fill: currentColor !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--dm-chat-attachment-text) !important;
}

[data-testid="stChatInput"] [data-testid="stUploadedFile"] > div:first-child,
[data-testid="stChatInput"] [data-testid="stFileUploaderFile"] > div:first-child,
[data-testid="stChatInput"] [data-testid="stChatInputFile"] > div:first-child {
    background: var(--dm-accent-soft) !important;
    border: 1px solid var(--dm-chat-attachment-border) !important;
    color: var(--dm-chat-icon) !important;
}

[data-testid="stChatInput"] [data-testid="stUploadedFile"] svg,
[data-testid="stChatInput"] [data-testid="stFileUploaderFile"] svg,
[data-testid="stChatInput"] [data-testid="stChatInputFile"] svg,
[data-testid="stChatInput"] [data-testid="stUploadedFile"] img,
[data-testid="stChatInput"] [data-testid="stFileUploaderFile"] img,
[data-testid="stChatInput"] [data-testid="stChatInputFile"] img,
[data-testid="stChatInput"] [data-testid="stUploadedFile"] [data-testid="stIconMaterial"],
[data-testid="stChatInput"] [data-testid="stFileUploaderFile"] [data-testid="stIconMaterial"],
[data-testid="stChatInput"] [data-testid="stChatInputFile"] [data-testid="stIconMaterial"] {
    color: var(--dm-chat-icon) !important;
    -webkit-text-fill-color: var(--dm-chat-icon) !important;
}

[data-testid="stChatInput"] [data-testid="stUploadedFile"] small,
[data-testid="stChatInput"] [data-testid="stFileUploaderFile"] small,
[data-testid="stChatInput"] [data-testid="stChatInputFile"] small,
[data-testid="stChatInput"] [data-testid="stUploadedFile"] [data-testid="stUploadedFileSize"],
[data-testid="stChatInput"] [data-testid="stFileUploaderFile"] [data-testid="stUploadedFileSize"],
[data-testid="stChatInput"] [data-testid="stChatInputFile"] [data-testid="stUploadedFileSize"] {
    color: var(--dm-chat-attachment-muted) !important;
    -webkit-text-fill-color: var(--dm-chat-attachment-muted) !important;
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
    align-items: center !important;
    background: var(--dm-panel);
    border: 1px solid var(--dm-border);
    border-radius: 0.85rem;
    box-sizing: border-box;
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 0.18rem !important;
    justify-content: flex-start !important;
    padding: 0.3rem 0.38rem;
    width: 100%;
}

[data-testid="stRadio"] [role="radiogroup"] label,
[data-testid="stRadio"] [role="radiogroup"] label div,
[data-testid="stRadio"] [role="radiogroup"] label span,
[data-testid="stRadio"] [role="radiogroup"] label p {
    color: var(--dm-muted) !important;
    font-size: 0.9rem !important;
    line-height: 1.35 !important;
    opacity: 1 !important;
    white-space: nowrap !important;
    -webkit-text-fill-color: var(--dm-muted) !important;
}

[data-testid="stRadio"] [role="radiogroup"] label {
    align-items: center !important;
    display: inline-flex !important;
    flex: 0 0 auto !important;
    gap: 0.18rem !important;
    margin: 0 !important;
    min-width: 0 !important;
    padding: 0.12rem 0.16rem !important;
}

[data-testid="stRadio"] [role="radiogroup"] label > div {
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked),
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) div,
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) span,
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {
    color: var(--dm-text) !important;
    font-weight: 700;
    -webkit-text-fill-color: var(--dm-text) !important;
}

[data-testid="stRadio"] input[type="radio"] {
    accent-color: var(--dm-accent) !important;
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
