import re

SYS_POLICY = """You are a grounded assistant.
- For latest/official/docs/time-sensitive queries, you MUST call TavilySearch first, then answer concisely with citations.
- If the user asks to save/export to txt (예: '저장', 'txt로 저장', 'save this'):
  1) First, compose the FULL final response you would show to the user (or, if they said "save the previous answer", use the MOST RECENT assistant message).
  2) Then call the save_text tool ONCE with that exact final response in 'content'. Use 'filename_prefix' if provided.
  3) After the tool returns, acknowledge the saved filename and DO NOT call save_text again.
"""

NEED_SEARCH_PATTERNS = [
    r"\b(latest|official|docs?|documentation|reference)\b",
    r"(최신|공식|문서|레퍼런스|가격|발매|지원 버전|변경점|로드맵)"
]

# NEW: detect save/export intent
NEED_SAVE_PATTERNS = [
    r"\b(save|export|write|txt)\b",
    r"(저장|내보내|텍스트|txt로|파일로)"
]

def needs_search(text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in NEED_SEARCH_PATTERNS)

def needs_save(text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in NEED_SAVE_PATTERNS)
