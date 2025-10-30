import re

SYS_POLICY = """You are a grounded assistant.
- For queries asking for latest/official docs or anything time-sensitive, you MUST call TavilySearch first, then answer with concise bullets and cite sources.
"""

NEED_SEARCH_PATTERNS = [
    r"\b(latest|official|docs?|documentation|reference)\b",
    r"(최신|공식|문서|레퍼런스|가격|발매|지원 버전|변경점|로드맵)"
]

def needs_search(text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in NEED_SEARCH_PATTERNS)
