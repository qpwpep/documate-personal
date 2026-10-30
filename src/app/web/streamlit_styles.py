from __future__ import annotations

from html import escape

import streamlit as st

from src.app.web.streamlit_theme import build_theme_css


_COMPONENT_CSS = """
#MainMenu,
footer {
    visibility: hidden;
}

[data-testid="stAppDeployButton"] {
    display: none;
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
    color: var(--dm-text);
}

/* Streamlit translates the collapsed sidebar off screen without removing its
   children from keyboard navigation. Keep only the separate opener available. */
[data-testid="stSidebar"][data-collapsed="true"],
[data-testid="stSidebar"][data-collapsed="true"] * {
    visibility: hidden !important;
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

.dm-conversation-title {
    display: block;
    margin-top: 0.35rem;
    overflow-wrap: anywhere;
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
    align-items: center;
    border-bottom: 1px solid var(--dm-strong-divider);
    display: flex;
    gap: 0.65rem;
    margin-bottom: 0.35rem;
    padding-bottom: 0.6rem;
}

.dm-sidebar-brand .dm-mark {
    flex-shrink: 0;
}

.dm-sidebar-brand h2 {
    color: var(--dm-text);
    font-size: 1.12rem;
    letter-spacing: 0;
    line-height: 1.3;
    margin: 0;
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
    overflow-wrap: anywhere;
}

.dm-upload-summary {
    color: var(--dm-text);
    margin: 0;
}

.dm-upload-help {
    color: var(--dm-muted);
    line-height: 1.55;
    overflow-wrap: anywhere;
}

[data-testid="stSidebar"] div.stButton > button {
    min-height: 3rem;
}

.st-key-documate_recent_conversations button[kind="primary"]:disabled {
    background: var(--dm-accent-soft);
    border-color: var(--dm-accent);
    color: var(--dm-text);
    opacity: 1;
}

.st-key-documate_recent_conversations button p {
    overflow-wrap: anywhere;
    white-space: normal;
}

[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(div.stButton),
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]) {
    margin-bottom: 0;
}

[data-testid="stSidebar"] [data-testid="stRadio"] {
    margin: 0;
}

[data-testid="stSidebar"] .dm-slack-feedback {
    color: var(--dm-muted);
    font-size: 0.875rem;
    line-height: 1.5;
    margin: 0;
    overflow-wrap: anywhere;
}

[data-testid="stSidebar"] .dm-slack-feedback[data-slack-invalid="true"] {
    background: var(--dm-error-bg);
    border-radius: 0.5rem;
    color: var(--dm-error-text);
    padding: 0.65rem;
}

.dm-intro {
    margin: 0.5rem auto;
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
    font-size: 2.25rem;
    letter-spacing: 0;
    line-height: 1.12;
    margin: 0;
    padding: 0;
}

.dm-intro p {
    color: var(--dm-muted);
    font-size: 1.06rem;
    line-height: 1.7;
    margin: 0.5rem auto;
    max-width: 610px;
}

.dm-docs-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin: 0.6rem auto 1.15rem;
}

/* The initial guide should not scroll out of view just to reserve space for
   a long chat history. Streamlit already allocates room for the composer. */
.block-container:has(.dm-intro) {
    padding-bottom: 1rem;
}

.dm-docs-row span {
    background: var(--dm-panel);
    border: 1px solid var(--dm-border);
    border-radius: 999px;
    color: var(--dm-muted);
    font-size: 0.82rem;
    padding: 0.34rem 0.7rem;
}

div.stButton > button,
div.stLinkButton > a,
[class*="-stop_response"] button {
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

div.stButton > button:hover,
div.stLinkButton > a:hover,
[class*="-stop_response"] button:hover {
    background: var(--dm-button-hover-bg);
    border-color: var(--dm-button-hover-border);
    color: var(--dm-text);
    transform: translateY(-1px);
}

div.stButton > button:focus,
div.stLinkButton > a:focus-visible,
[class*="-stop_response"] button:focus-visible {
    border-color: var(--dm-accent);
    box-shadow: 0 0 0 0.14rem var(--dm-focus-ring);
}

/* The interruption button's help tooltip inserts a wrapper between stButton
   and button. Pair its native label with the independent DocuMate theme. */
[class*="-stop_response"] button * {
    color: var(--dm-text) !important;
    -webkit-text-fill-color: var(--dm-text) !important;
}

a[href^="#dm-citation-"]:focus-visible {
    border-radius: 0.2rem;
    outline: 2px solid var(--dm-accent);
    outline-offset: 3px;
}

/* Recommendation text is the question itself, so it must remain readable on
   narrow screens instead of using Streamlit's single-line button ellipsis. */
[class*="-selected_upload_prompts"]:not([data-selected-files="true"]),
[class*="-selected_upload_prompts"][hidden],
[class*="-default_prompts"][hidden] {
    display: none !important;
}

[class*="-quick_prompt_"] button [data-testid="stMarkdownContainer"],
[class*="-quick_prompt_"] button [data-testid="stMarkdownContainer"] p {
    overflow: visible;
    text-overflow: clip;
    white-space: normal;
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

[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
    background: var(--dm-accent-soft);
    color: var(--dm-accent);
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

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    backdrop-filter: blur(10px);
    background: var(--dm-assistant-bg);
    border: 1px solid var(--dm-assistant-border);
    border-radius: 1.1rem;
    box-shadow: var(--dm-assistant-shadow);
    color: var(--dm-text);
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

[data-testid="stCode"] pre {
    background: var(--dm-code-bg) !important;
    color: var(--dm-inline-code-text) !important;
}

[data-testid="stCode"] pre code {
    background: transparent !important;
    color: var(--dm-inline-code-text) !important;
}

/* Prism's own theme does not follow the app preference. Keep its token colors
   and an opaque background together so every code token stays readable. */
[data-testid="stCode"] .token {
    background: transparent !important;
    color: var(--dm-inline-code-text) !important;
}

[data-testid="stCode"] .token:is(.comment, .prolog, .doctype, .cdata, .punctuation, .unchanged),
[data-testid="stCode"] .comment.linenumber,
[data-testid="stCode"] .react-syntax-highlighter-line-number {
    color: var(--dm-muted) !important;
}

[data-testid="stCode"] .token:is(.keyword, .tag, .function, .class-name, .selector, .property, .attr-name, .variable) {
    color: var(--dm-code-keyword) !important;
}

[data-testid="stCode"] .token:is(.string, .char, .attr-value, .inserted) {
    color: var(--dm-code-string) !important;
}

[data-testid="stCode"] .token:is(.boolean, .constant, .symbol, .number, .regex, .url) {
    color: var(--dm-code-value) !important;
}

[data-testid="stCode"] .token:is(.operator, .entity, .decorator, .atrule) {
    color: var(--dm-code-operator) !important;
}

[data-testid="stCode"] .token:is(.important, .deleted) {
    color: var(--dm-code-important) !important;
}

[data-testid="stCode"] .namespace {
    opacity: 1;
}

[data-testid="stCodeCopyButton"] {
    color: var(--dm-muted);
}

[data-testid="stCodeCopyButton"]:hover {
    background: var(--dm-accent-soft);
    color: var(--dm-text);
}

[data-testid="stPopoverButton"],
[data-testid="stPopoverBody"] {
    background: var(--dm-panel);
    border-color: var(--dm-border);
    color: var(--dm-text);
}

[data-testid="stPopoverBody"] {
    box-shadow: var(--dm-shadow);
}

.dm-source-title {
    margin: 0;
    overflow-wrap: anywhere;
}

.dm-source-header {
    align-items: center;
    background: var(--dm-panel);
    border-bottom: 1px solid var(--dm-divider);
    display: flex;
    gap: 0.75rem;
    justify-content: space-between;
    padding: 0.25rem 0 0.65rem;
}

[data-testid="stPopoverBody"] [data-testid="stElementContainer"]:has(.dm-source-header) {
    background: var(--dm-panel);
    position: sticky;
    top: 0;
    z-index: 2;
}

.dm-source-close {
    background: var(--dm-panel);
    border: 1px solid var(--dm-border);
    border-radius: 0.6rem;
    color: var(--dm-text);
    cursor: pointer;
    flex-shrink: 0;
    font: inherit;
    min-height: 44px;
    min-width: 44px;
    padding: 0.5rem 0.75rem;
}

.dm-source-close:hover {
    background: var(--dm-accent-soft);
}

.dm-source-close:focus-visible,
.dm-source-title:focus-visible {
    outline: 2px solid var(--dm-accent);
    outline-offset: 2px;
}

.dm-source-answer-context {
    background: var(--dm-accent-soft);
    border-left: 3px solid var(--dm-accent);
    border-radius: 0.3rem;
    color: var(--dm-text);
    margin: 0.75rem 0;
    overflow-wrap: anywhere;
    padding: 0.75rem;
}

.dm-source-context-label {
    color: var(--dm-accent);
    font-size: 0.88rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.dm-source-answer-text {
    margin: 0.4rem 0 0;
}

[data-testid="stPopoverButton"]:hover {
    background: var(--dm-button-hover-bg);
    border-color: var(--dm-button-hover-border);
    color: var(--dm-text);
}

[data-testid="stPopoverButton"]:focus {
    border-color: var(--dm-accent);
    color: var(--dm-text);
}

[data-testid="stPopoverBody"] > div {
    background: var(--dm-panel);
    color: var(--dm-text);
}

[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] {
    color: var(--dm-text);
}

/* Plain retained source uses st.text rather than Markdown. Its inner span has
   a Streamlit theme color of its own, including when the app theme changes. */
[data-testid="stText"],
[data-testid="stText"] * {
    color: var(--dm-text) !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] [data-testid="stMarkdownContainer"] {
    color: var(--dm-muted);
    opacity: 1;
}

[data-testid="stChatMessage"] a,
[data-testid="stPopoverBody"] a {
    color: var(--dm-accent);
}

[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] td,
[data-testid="stPopoverBody"] th,
[data-testid="stPopoverBody"] td {
    border-color: var(--dm-border);
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stAlert"] {
    backdrop-filter: blur(10px);
    background: var(--dm-assistant-bg) !important;
    border: 1px solid var(--dm-assistant-border) !important;
    border-radius: 1.1rem !important;
    box-shadow: var(--dm-assistant-shadow);
    color: var(--dm-text) !important;
}

/* Streamlit's status colors follow its own theme. Opaque paired colors keep
   status messages readable when DocuMate's independent theme changes. */
[data-testid="stAlertContainer"] {
    --dm-alert-text: var(--dm-info-text);
    --dm-alert-bg: var(--dm-info-bg);
    background: var(--dm-alert-bg) !important;
    color: var(--dm-alert-text) !important;
}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
    --dm-alert-text: var(--dm-success-text);
    --dm-alert-bg: var(--dm-success-bg);
}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
    --dm-alert-text: var(--dm-warning-text);
    --dm-alert-bg: var(--dm-warning-bg);
}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
    --dm-alert-text: var(--dm-error-text);
    --dm-alert-bg: var(--dm-error-bg);
}

[data-testid="stAlertContainer"] * {
    color: inherit !important;
}

[data-testid="stAlertContainer"] a {
    text-decoration: underline;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stAlert"] code {
    background: var(--dm-inline-code-bg) !important;
    border: 1px solid var(--dm-inline-code-border);
    color: var(--dm-inline-code-text) !important;
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

#documate-upload-selection {
    background: var(--dm-chat-input-bg);
    border: 1px solid var(--dm-border);
    border-radius: 0.65rem;
    color: var(--dm-text);
    font-size: 0.875rem;
    line-height: 1.5;
    margin: 0 0 0.5rem;
    overflow-wrap: anywhere;
    padding: 0.6rem 0.75rem;
}

#documate-upload-selection:empty {
    display: none;
}

#documate-upload-selection[data-invalid="true"] {
    border-color: var(--dm-accent);
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

[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"],
[data-testid="stChatInput"] button:has(svg):not([aria-label^="Remove "]) {
    min-width: 44px;
    min-height: 44px;
    flex-shrink: 0;
}

[data-testid="stChatInput"] button:focus-visible {
    outline: 2px solid var(--dm-accent);
    outline-offset: 2px;
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
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] p {
    color: var(--dm-text) !important;
}

[data-testid="stSelectbox"] [role="group"],
[data-testid="stSelectbox"] div:has(> input[role="combobox"]),
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] [role="listbox"] {
    background: var(--dm-panel) !important;
    border-color: var(--dm-border) !important;
    color: var(--dm-text) !important;
}

[data-testid="stSelectbox"] input,
[data-testid="stSelectbox"] button,
[data-testid="stSelectboxVirtualDropdown"] [role="option"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"] * {
    color: var(--dm-text) !important;
    -webkit-text-fill-color: var(--dm-text) !important;
}

[data-testid="stSelectbox"] input,
[data-testid="stSelectbox"] button {
    background: transparent !important;
}

[data-testid="stTextInput"] input:disabled,
[data-testid="stSelectbox"] input:disabled {
    color: var(--dm-muted) !important;
    -webkit-text-fill-color: var(--dm-muted) !important;
}

[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"] {
    background: var(--dm-accent-soft) !important;
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

.stAlert {
    border-radius: 0.85rem;
}

@media (max-width: 760px) {
    [data-testid="stPopoverBody"]:has(.dm-source-title) {
        /* Keep the expanded reader inside the viewport; the popover's computed
           transform still assumes Streamlit's shorter default height. */
        top: 4rem !important;
        left: 1rem !important;
        transform: none !important;
        width: calc(100vw - 2rem) !important;
        min-width: 0 !important;
        max-width: calc(100vw - 2rem);
        max-height: calc(100dvh - 5rem) !important;
    }

    /* Reserve a separate toolbar row even after the conversation is scrolled. */
    header[data-testid="stHeader"] {
        height: 3.75rem;
        background: var(--dm-bg);
        border-bottom: 1px solid var(--dm-border);
    }

    #stMain {
        top: 3.75rem;
        height: calc(100% - 3.75rem);
    }

    [data-testid="stToolbar"] {
        padding-right: 0.75rem;
    }

    .block-container {
        padding: 1rem 0.75rem 7.5rem;
    }

    .dm-topbar {
        align-items: flex-start;
        flex-direction: column;
    }

    .dm-topbar-status {
        white-space: normal;
    }

    .dm-intro {
        margin-top: 1rem;
        text-align: left;
    }

    .dm-intro h1 {
        font-size: 1.8rem;
    }

    .dm-docs-row {
        justify-content: flex-start;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        max-width: 100%;
    }

    /* Put the speaker above the message so its column cannot consume the
       narrow reading area. Keep the avatar and the message's accessible name. */
    [data-testid="stChatMessage"] {
        flex-direction: column;
        gap: 0.35rem;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        width: 100%;
        min-width: 0;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
        padding: 0.7rem 0.75rem;
    }

    [data-testid="stChatMessage"] h2 {
        font-size: 1.5rem;
        line-height: 1.35;
        padding-top: 0.6rem;
    }

    [data-testid="stChatMessage"] h3 {
        font-size: 1.2rem;
    }

    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] {
        padding: 0.65rem 0.75rem;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stAlert"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none;
        padding: 0;
    }

    [data-testid="stChatMessage"] [data-testid="stAlertContainer"] {
        padding: 0.65rem 0.75rem;
    }

    [data-testid="stChatMessage"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
        background: transparent;
        border: 0;
        box-shadow: none;
        padding: 0;
    }
}
"""


def configure_page(theme_mode: str) -> None:
    st.set_page_config(page_title="DocuMate", layout="wide")
    st.markdown(
        f"<style>\n{build_theme_css(theme_mode)}\n{_COMPONENT_CSS}</style>",
        unsafe_allow_html=True,
    )


def render_conversation_header(
    title: str, *, attachment_count: int | None, read_only: bool, attachment_error: bool = False,
) -> None:
    if attachment_error:
        status = "첨부 준비 실패"
        if attachment_count is not None:
            status += f" · 기존 자료 {attachment_count}개 유지"
    elif attachment_count is None:
        status = "첨부 연결 확인 중"
    else:
        status = f"{'이전 첨부 기록' if read_only else '첨부 자료'} {attachment_count}개"
    if read_only:
        status = "읽기 전용 · " + status
    st.markdown(
        f"""
        <div class="dm-topbar">
            <div class="dm-topbar-title">
                <span class="dm-mark">DM</span>
                <div class="dm-title-copy">
                    <strong>DocuMate</strong>
                    <span class="dm-conversation-title">{escape(title)}</span>
                </div>
            </div>
            <span class="dm-topbar-status">{status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
