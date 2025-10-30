import re

SYS_POLICY = """"당신은 공식 문서(official docs) 검색 결과만을 근거로 답하는 조수입니다.\n"
"각 항목은 간결하게 정리하고, 모든 주장에 대해 [◆ 출처]에 해당 문서의 정확한 URL을 명시하세요.\n"
"공식 문서 출처가 없으면 그 사실을 밝히고, 필요 시 추가 검색을 제안하세요."
"""

NEED_SEARCH_PATTERNS = [
    r"\b(latest|official|docs?|documentation|reference)\b",
    r"(최신|공식|문서|레퍼런스|가격|발매|지원 버전|변경점|로드맵)"
]

def needs_search(text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in NEED_SEARCH_PATTERNS)
