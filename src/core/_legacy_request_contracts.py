from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.core.rules import get_rules_config


class AnswerContract(BaseModel):
    """Question requirements, independent of retrieval sources and visual sections."""

    model_config = ConfigDict(extra="forbid")
    code_example: bool = False
    ordered_steps: bool = False
    options_summary: bool = False
    comparison: bool = False
    checklist: bool = False


def _has_any(query: str, *markers: str) -> bool:
    lowered = str(query or "").lower()
    return any(marker.lower() in lowered for marker in markers)


def _has_explicit_comparison(query: str) -> bool:
    if re.search(r"\b(?:compare|comparison|versus|vs)\b", query, flags=re.I):
        return True
    try:
        return re.search(get_rules_config().planner.compare_clause_pattern, query) is not None
    except re.error:
        return False


def is_code_example_request(query: str) -> bool:
    return _has_any(
        query, "code example", "example code", "sample code", "code sample",
        "예제", "예시", "샘플 코드", "샘플코드", "코드 샘플", "코드샘플",
    )


def is_options_summary_request(query: str) -> bool:
    return _has_any(
        query, "option", "parameter", "argument", "옵션", "파라미터", "매개변수", "인자",
    )


def infer_answer_contract(query: str) -> AnswerContract:
    return AnswerContract(
        code_example=is_code_example_request(query),
        ordered_steps=_has_any(query, "단계별", "step by step", "초보자"),
        options_summary=is_options_summary_request(query),
        comparison=_has_explicit_comparison(str(query or "")),
        checklist=_has_any(query, "체크리스트", "checklist"),
    )


def missing_required_content(contract: AnswerContract, document: Any) -> list[str]:
    """Check observable form only; semantic comparison requires separate evaluation."""
    blocks = document.get("blocks", []) if isinstance(document, dict) else getattr(document, "blocks", [])
    block_data = [item if isinstance(item, dict) else item.model_dump() for item in blocks]
    missing: list[str] = []
    if contract.code_example and not any(block.get("type") == "code" for block in block_data):
        missing.append("code_example")
    if contract.ordered_steps and not any(
        block.get("type") == "list" and block.get("ordered") for block in block_data
    ):
        missing.append("ordered_steps")
    if contract.checklist and not any(block.get("type") == "list" for block in block_data):
        missing.append("checklist")
    return missing
