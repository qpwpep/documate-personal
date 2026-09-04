import re

from src.core.rules import get_rules_config


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
- If the user asks to save or share without reusable prior answer context, generate a self-contained final body in this turn instead of asking what to save/share.
- Keep answers grounded in retrieved evidence when evidence is available.
"""

_ASCII_IDENTIFIER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9._-]{1,}(?![A-Za-z0-9_])")
_TECHNICAL_EXPLAINER_PATTERNS = (
    r"\b(explain|overview|intro(?:duction)?|usage|how\s+to\s+use|guide|tutorial|best practice|performance|optimi[sz]ation)\b",
    r"(알려줘|설명(?:해줘)?|소개(?:해줘)?|개요|기본\s*사용법|사용법|문법|파라미터|매개변수|옵션|예제|예시|성능\s*최적화|최적화|가이드|튜토리얼)",
)
_TOPIC_PLUS_TECHNICAL_REQUEST_PATTERNS = (
    r"[A-Za-z][A-Za-z0-9._-]{1,}\s*(?:에\s*대해|의)?\s*(?:알려줘|설명(?:해줘)?|소개(?:해줘)?|개요|기본\s*사용법|사용법|문법|파라미터|매개변수|옵션|예제|예시|성능\s*최적화|최적화|가이드|튜토리얼)",
    r"[가-힣A-Za-z0-9._-]{2,}(?:에\s*대해|의)\s*(?:알려줘|설명(?:해줘)?|소개(?:해줘)?|개요|기본\s*사용법|사용법|문법|파라미터|매개변수|옵션|예제|예시|성능\s*최적화|최적화|가이드|튜토리얼)",
)


def _matches_any(text: str, patterns: list[str]) -> bool:
    candidate = str(text or "")
    return any(re.search(pattern, candidate, flags=re.I) for pattern in patterns)


def _looks_like_docs_explainer_request(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False

    if _matches_any(candidate, list(_TOPIC_PLUS_TECHNICAL_REQUEST_PATTERNS)):
        return True

    if _ASCII_IDENTIFIER_PATTERN.search(candidate) and _matches_any(candidate, list(_TECHNICAL_EXPLAINER_PATTERNS)):
        return True

    return False


def needs_search(text: str) -> bool:
    return _matches_any(text, get_rules_config().intents.docs_patterns) or _looks_like_docs_explainer_request(text)


def needs_save(text: str) -> bool:
    return _matches_any(text, get_rules_config().intents.save_patterns)


def needs_slack(text: str) -> bool:
    return _matches_any(text, get_rules_config().intents.slack_patterns)
