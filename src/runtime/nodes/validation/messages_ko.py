from __future__ import annotations

import re


_SECTION_HEADINGS = {
    "summary": "요약",
    "checklist": "체크리스트",
    "steps": "단계별 안내",
    "official_docs": "공식 문서",
    "comparison": "비교",
    "interpretation_a": "해석 A",
    "interpretation_b": "해석 B",
}
_ROUTE_PREFIXES = {
    "docs": "공식 문서 기준으로",
    "upload": "업로드 파일에서는",
    "local": "로컬 자료에서는",
}
_HYBRID_DOCS_PREFIX = "공식 문서 기준으로는"
_HYBRID_LOCAL_PREFIXES = {
    "upload": "반면 업로드 파일에서는",
    "local": "반면 로컬 자료에서는",
}
ROUTE_PREFIX_PATTERN = re.compile(
    r"^(?:공식 문서 기준으로(?:는)?|(?:반면\s+)?업로드 파일에서는|(?:반면\s+)?로컬 자료에서는|(?:반면\s+)?업로드 예시에서는)\s+"
)


def section_heading(kind: str, *, required_routes: set[str]) -> str:
    if kind == "upload_code":
        return "업로드 코드" if "upload" in required_routes else "로컬 코드"
    return _SECTION_HEADINGS.get(str(kind or "").strip(), str(kind or "").strip())


def route_prefix(route: str) -> str:
    return _ROUTE_PREFIXES.get(str(route or "").strip(), "")


def hybrid_docs_prefix() -> str:
    return _HYBRID_DOCS_PREFIX


def hybrid_local_prefix(route: str) -> str:
    return _HYBRID_LOCAL_PREFIXES.get(str(route or "").strip(), _HYBRID_LOCAL_PREFIXES["local"])


def hybrid_limit_sentence(local_route: str) -> str:
    if local_route == "upload":
        return "근거는 공식 문서 1건과 업로드 파일 1건만 반영했습니다."
    return "근거는 공식 문서 1건과 로컬 자료 1건만 반영했습니다."


__all__ = [
    "ROUTE_PREFIX_PATTERN",
    "hybrid_docs_prefix",
    "hybrid_limit_sentence",
    "hybrid_local_prefix",
    "route_prefix",
    "section_heading",
]
