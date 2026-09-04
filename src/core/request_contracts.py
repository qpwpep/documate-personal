from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.rules import get_rules_config

SectionKind = Literal[
    "summary",
    "checklist",
    "steps",
    "code_example",
    "options",
    "official_docs",
    "upload_code",
    "comparison",
    "interpretation_a",
    "interpretation_b",
]


class AnswerContract(BaseModel):
    required_sections: list[SectionKind] = Field(default_factory=list)
    ordered_steps: bool = False
    split_by_source: bool = False


def _has_any(query: str, *markers: str) -> bool:
    lowered = str(query or "").lower()
    return any(marker.lower() in lowered for marker in markers)


def _has_explicit_comparison(query: str) -> bool:
    if _has_any(query, "compare", "comparison", "versus", "vs", "vs."):
        return True
    try:
        return re.search(get_rules_config().planner.compare_clause_pattern, str(query or "")) is not None
    except re.error:
        return False


def is_code_example_request(query: str) -> bool:
    return _has_any(
        query,
        "code example",
        "code examples",
        "example code",
        "sample code",
        "code sample",
        "code samples",
        "예제",
        "예시",
        "코드 예제",
        "코드예제",
        "샘플 코드",
        "샘플코드",
        "코드 샘플",
        "코드샘플",
    )


def is_options_summary_request(query: str) -> bool:
    return _has_any(
        query,
        "option",
        "options",
        "parameter",
        "parameters",
        "argument",
        "arguments",
        "옵션",
        "파라미터",
        "매개변수",
        "인자",
    )


def infer_answer_contract(query: str, required_routes: list[str] | None = None) -> AnswerContract:
    required_sections: list[SectionKind] = []
    if _has_any(query, "요약", "summary"):
        required_sections.append("summary")
    if _has_any(query, "체크리스트", "checklist"):
        required_sections.append("checklist")
    if _has_any(query, "단계별", "step by step", "초보자"):
        required_sections.append("steps")
    if is_code_example_request(query):
        required_sections.append("code_example")
    if _has_any(query, "가능한 해석 2가지", "해석 2가지", "two interpretations"):
        required_sections.extend(["interpretation_a", "interpretation_b"])

    route_set = {str(route or "").strip() for route in (required_routes or []) if str(route or "").strip()}
    split_by_source = {"docs", "upload"}.issubset(route_set)
    if "docs" in route_set and not split_by_source and is_options_summary_request(query):
        required_sections.append("options")
    if split_by_source:
        if _has_explicit_comparison(query):
            required_sections.extend(["official_docs", "upload_code", "comparison"])
        else:
            required_sections.extend(["official_docs", "comparison"])

    deduped_sections: list[SectionKind] = []
    seen: set[str] = set()
    for section in required_sections:
        if section not in seen:
            deduped_sections.append(section)
            seen.add(section)

    return AnswerContract(
        required_sections=deduped_sections,
        ordered_steps=_has_any(query, "단계별", "step by step", "초보자"),
        split_by_source=split_by_source,
    )


def render_answer_contract_prompt(contract: AnswerContract) -> str:
    lines = ["[Answer Contract]"]
    lines.append(
        "- required_sections=" + (", ".join(contract.required_sections) if contract.required_sections else "none")
    )
    lines.append(f"- ordered_steps={str(contract.ordered_steps).lower()}")
    lines.append(f"- split_by_source={str(contract.split_by_source).lower()}")
    lines.append("- Return `sections` whose `kind` values exactly match the required sections when they are requested.")
    lines.append("- Do not omit required sections.")
    if "code_example" in contract.required_sections:
        lines.append("- The code_example section must include at least one fenced code block with runnable sample code.")
    if "options" in contract.required_sections:
        lines.append("- The options section must group API parameters/options into concise bullets by purpose.")
        lines.append("- Prefer exact parameter names from Retrieved Evidence candidate_facts or doc_metadata.")
    if contract.split_by_source:
        lines.append("- For hybrid compare tasks, keep official docs facts and uploaded code facts separate before writing the comparison.")
        if "upload_code" in contract.required_sections:
            lines.append("- Keep official_docs, upload_code, and comparison section bodies to 2-3 short sentences each.")
        else:
            lines.append("- Keep official_docs to 2-3 short sentences and comparison to 1-2 short sentences.")
            lines.append("- Include uploaded code details inside comparison in one concrete sentence.")
        lines.append("- Use at most 4 total claims for hybrid answers; prefer 3 claims when possible.")
        lines.append("- Keep the comparison to 1-2 sentences about the official-docs match or difference.")
    return "\n".join(lines)


def _iter_present_section_kinds(output: Any) -> set[str]:
    if output is None:
        return set()
    sections = getattr(output, "sections", None)
    if sections is None and isinstance(output, dict):
        sections = output.get("sections", [])
    if not isinstance(sections, list):
        return set()
    present: set[str] = set()
    for section in sections:
        kind = ""
        if isinstance(section, dict):
            kind = str(section.get("kind") or "").strip()
        else:
            kind = str(getattr(section, "kind", "") or "").strip()
        if kind:
            present.add(kind)
    return present


def missing_required_sections(contract: AnswerContract, output: Any) -> list[str]:
    present = _iter_present_section_kinds(output)
    return [section for section in contract.required_sections if section not in present]
