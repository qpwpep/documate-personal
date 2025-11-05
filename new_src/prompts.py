import re

# ============================================================
# 🧭 SYSTEM POLICY — Unified for TavilySearch, RAGSearch, SaveText
# ============================================================
SYS_POLICY = """당신은 세 가지 주요 능력을 가진 조수입니다:

1️⃣ TavilySearch — 공식 문서 기반 검색  
   - 개념, 문법, API, 매개변수 등 **최신 공식 정보**가 필요한 경우 사용하세요.  
   - 결과에는 반드시 [◆ 공식 문서]와 함께 URL을 명시합니다.

2️⃣ RAGSearch — 로컬 노트북 기반 검색  
   - 코드 예제, 프로젝트, 실습 기반 사용 사례가 필요한 경우 사용하세요.  
   - 결과에는 [◆ 로컬 예제]와 함께 노트북 경로를 명시합니다.  
   - TavilySearch 결과와 함께 사용할 수 있습니다(예: 개념 + 예제 통합 답변).

3️⃣ SaveText — 응답을 텍스트 파일(.txt)로 저장  
   - 사용자가 "저장", "txt로 저장", "save this" 등으로 요청하면 다음을 따르세요:
     ① 사용자에게 보여줄 **최종 완성된 응답 전체**를 `content`에 담아 `save_text` 도구를 한 번만 호출합니다.  
     ② `filename_prefix`가 주어졌다면 함께 전달합니다.  
     ③ 도구가 반환한 파일명(`Saved output to ...`)을 짧게 확인(acknowledge)하고,  
        같은 턴에서는 다시 save_text를 호출하지 않습니다.

💡 응답 규칙:
- 질문이 개념 중심이면 TavilySearch →  
  예제 중심이면 RAGSearch →  
  둘 다 필요하면 TavilySearch → RAGSearch 순으로 사용합니다.  
- 가능한 한 두 결과를 **자연스럽게 통합하여 설명**하고,  
  각각의 출처를 [◆ 공식 문서], [◆ 로컬 예제]로 구분해 명시하세요.
"""

# ============================================================
# 🔍 PATTERN MATCHING FOR TOOL DECISIONS
# ============================================================

# Official docs / TavilySearch trigger
NEED_SEARCH_PATTERNS = [
    r"\b(latest|official|docs?|documentation|reference|api|syntax|parameter|manual)\b",
    r"(최신|공식|문서|레퍼런스|함수|매개변수|사용법|방법|API)"
]

# Local notebooks / RAGSearch trigger
NEED_RAG_PATTERNS = [
    r"\b(example|sample|notebook|project|code|implementation|practice)\b",
    r"(이전|노트북|예제|코드|실습|프로젝트|데이터셋|baseline|결과)"
]

# SaveText trigger
NEED_SAVE_PATTERNS = [
    r"\b(save|export|write|txt)\b",
    r"(저장|내보내|텍스트|txt로|파일로)"
]

# ============================================================
# 🧩 Detection helpers
# ============================================================

def needs_search(text: str) -> bool:
    """Return True if text implies official-doc search."""
    return any(re.search(p, text, flags=re.I) for p in NEED_SEARCH_PATTERNS)

def needs_rag(text: str) -> bool:
    """Return True if text implies local-notebook (RAG) search."""
    return any(re.search(p, text, flags=re.I) for p in NEED_RAG_PATTERNS)

def needs_save(text: str) -> bool:
    """Return True if text implies save/export request."""
    return any(re.search(p, text, flags=re.I) for p in NEED_SAVE_PATTERNS)