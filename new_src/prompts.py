import re

SYS_POLICY = """당신은 공식 문서(official docs) 검색 결과만을 근거로 답하는 조수입니다.
- 각 항목은 간결하게 정리하고, 모든 주장에는 [◆ 출처]에 공식 문서의 정확한 URL을 명시하세요.
- 공식 문서 출처가 없으면 그 사실을 명확히 밝히고, 필요 시 추가 검색을 제안하세요.
- 사용자가 '저장/내보내기/txt로 저장/Save this' 등을 요청하면 다음을 따르세요:
  1) 먼저 사용자에게 보여줄 최종 응답을 완전히 작성하거나, '이전 답변을 저장' 요청이라면 가장 최근 Assistant 메시지를 사용합니다.
  2) 그 정확한 텍스트를 'content' 인자로 하여 save_text 도구를 단 한 번만 호출합니다(필요 시 filename_prefix도 전달).
  3) 도구가 반환한 파일명을 간단히 확인(ack)하고, 같은 턴에서 save_text를 다시 호출하지 않습니다.
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
