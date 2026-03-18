import re

from .rules import get_rules_config


SYS_POLICY = """You are DocuMate, a retrieval-first assistant.

Available capabilities:
1) TavilySearch for official and current documentation.
2) RAGSearch for local notebook/project examples.
3) UploadSearch for the currently uploaded file only.
4) SaveText for saving the final answer as a .txt file.
5) SlackNotify for sending the final answer to Slack.

Rules:
- Prefer official docs when the user asks for docs, API usage, latest behavior, or references.
- Prefer UploadSearch when the user asks about the currently uploaded file.
- Prefer local RAG only when the user explicitly asks for local examples, notebooks, or project references.
- When the user asks to save or share, the content to save/share is the final answer you generate in this turn unless the user explicitly names another target.
- Keep answers grounded in retrieved evidence when evidence is available.
"""


def _matches_any(text: str, patterns: list[str]) -> bool:
    candidate = str(text or "")
    return any(re.search(pattern, candidate, flags=re.I) for pattern in patterns)


def needs_search(text: str) -> bool:
    return _matches_any(text, get_rules_config().intents.docs_patterns)


def has_explicit_docs_intent(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(keyword in lowered for keyword in get_rules_config().intents.explicit_docs_keywords)


def has_explicit_local_intent(text: str) -> bool:
    return _matches_any(text, get_rules_config().intents.local_patterns)


def needs_rag(text: str) -> bool:
    return has_explicit_local_intent(text) and not has_explicit_docs_intent(text)


def needs_save(text: str) -> bool:
    return _matches_any(text, get_rules_config().intents.save_patterns)


def needs_slack(text: str) -> bool:
    return _matches_any(text, get_rules_config().intents.slack_patterns)
