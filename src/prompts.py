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
- Prefer local RAG when the user asks for project/notebook examples.
- When the user asks to save or share, the content to save/share is the final answer you generate in this turn unless the user explicitly names another target.
- Keep answers grounded in retrieved evidence when evidence is available.
"""


NEED_SEARCH_PATTERNS = [
    r"\b(latest|official|docs?|documentation|reference|api|syntax|parameter|manual)\b",
    r"(최신|공식|문서|레퍼런스|참고자료|사용법|매개변수|파라미터|API)",
]

NEED_RAG_PATTERNS = [
    r"\b(example|sample|notebook|project|code|implementation|practice)\b",
    r"(예제|노트북|프로젝트|코드|구현|실습|샘플|baseline)",
]

NEED_SAVE_PATTERNS = [
    r"\b(save|export|write|download|txt|text file)\b",
    r"(저장|저장해|저장하고|내보내|파일로|텍스트 파일|txt로|다운로드)",
]

NEED_SLACK_PATTERNS = [
    r"\b(slack|dm|direct message|channel)\b",
    r"(슬랙|DM|디엠|채널).*(보내|전송|공유|전달)",
    r"(보내|전송|공유|전달).*(슬랙|slack|DM|디엠|채널)",
]


def needs_search(text: str) -> bool:
    """Return True if text implies official-doc search."""
    return any(re.search(p, text, flags=re.I) for p in NEED_SEARCH_PATTERNS)


def needs_rag(text: str) -> bool:
    """Return True if text implies local example lookup."""
    return any(re.search(p, text, flags=re.I) for p in NEED_RAG_PATTERNS)


def needs_save(text: str) -> bool:
    """Return True if text implies save/export request."""
    return any(re.search(p, text, flags=re.I) for p in NEED_SAVE_PATTERNS)


def needs_slack(text: str) -> bool:
    """Return True if text implies Slack share request."""
    return any(re.search(p, text, flags=re.I) for p in NEED_SLACK_PATTERNS)
