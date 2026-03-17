import re


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


_DOCS_PATTERNS = [
    r"\b(latest|official|docs?|documentation|reference|api|syntax|parameter|manual)\b",
    r"(\uacf5\uc2dd|\ubb38\uc11c|\ub808\ud37c\ub7f0\uc2a4|\ucc38\uace0\s*\uc790\ub8cc|\ucd5c\uc2e0|API)",
]
_EXPLICIT_DOCS_KEYWORDS = (
    "official",
    "docs",
    "documentation",
    "reference",
    "api",
    "manual",
    "\uacf5\uc2dd",
    "\ubb38\uc11c",
    "\ub808\ud37c\ub7f0\uc2a4",
    "\ucc38\uace0 \uc790\ub8cc",
    "\ucd5c\uc2e0",
)
_LOCAL_PATTERNS = [
    r"\b(example|sample|notebook|project|implementation|practice|baseline)\b",
    r"(\uc608\uc81c|\ub178\ud2b8\ubd81|\ud504\ub85c\uc81d\ud2b8|\uad6c\ud604|\uc2e4\uc2b5|\uc0d8\ud50c|\ubca0\uc774\uc2a4\ub77c\uc778)",
]
_SAVE_PATTERNS = [
    r"\b(save|export|write|download|txt|text file)\b",
    r"(\uc800\uc7a5|\ud14d\uc2a4\ud2b8\s*\ud30c\uc77c|txt\ub85c|\ub0b4\ubcf4\ub0b4|다운로드)",
]
_SLACK_PATTERNS = [
    r"\b(slack|dm|direct message|channel)\b",
    r"(\uc2ac\ub799|DM|\ucc44\ub110|\ub2e4\uc774\ub809\ud2b8\s*\uba54\uc2dc\uc9c0).*(\ubcf4\ub0b4|\uc804\uc1a1|\uacf5\uc720|\uc804\ub2ec)",
    r"(\ubcf4\ub0b4|\uc804\uc1a1|\uacf5\uc720|\uc804\ub2ec).*(\uc2ac\ub799|DM|\ucc44\ub110)",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    candidate = str(text or "")
    return any(re.search(pattern, candidate, flags=re.I) for pattern in patterns)


def needs_search(text: str) -> bool:
    """Return True if text implies official-doc search."""
    return _matches_any(text, _DOCS_PATTERNS)


def has_explicit_docs_intent(text: str) -> bool:
    """Return True if text explicitly asks for official docs/reference material."""
    lowered = str(text or "").lower()
    return any(keyword in lowered for keyword in _EXPLICIT_DOCS_KEYWORDS)


def has_explicit_local_intent(text: str) -> bool:
    """Return True if text explicitly asks for local examples/notebooks/projects."""
    return _matches_any(text, _LOCAL_PATTERNS)


def needs_rag(text: str) -> bool:
    """Return True if text implies local example lookup and does not explicitly ask for docs."""
    return has_explicit_local_intent(text) and not has_explicit_docs_intent(text)


def needs_save(text: str) -> bool:
    """Return True if text implies save/export request."""
    return _matches_any(text, _SAVE_PATTERNS)


def needs_slack(text: str) -> bool:
    """Return True if text implies Slack share request."""
    return _matches_any(text, _SLACK_PATTERNS)
