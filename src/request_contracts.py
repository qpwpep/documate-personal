from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SectionKind = Literal[
    "summary",
    "checklist",
    "steps",
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


def infer_answer_contract(query: str, required_routes: list[str] | None = None) -> AnswerContract:
    required_sections: list[SectionKind] = []
    if _has_any(query, "요약", "summary"):
        required_sections.append("summary")
    if _has_any(query, "체크리스트", "checklist"):
        required_sections.append("checklist")
    if _has_any(query, "단계별", "step by step", "초보자"):
        required_sections.append("steps")
    if _has_any(query, "가능한 해석 2가지", "해석 2가지", "two interpretations"):
        required_sections.extend(["interpretation_a", "interpretation_b"])

    split_by_source = set(required_routes or []) == {"docs", "upload"}
    if split_by_source:
        required_sections.extend(["official_docs", "upload_code", "comparison"])

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
    if contract.split_by_source:
        lines.append("- For hybrid compare tasks, keep official docs facts and uploaded code facts separate before writing the comparison.")
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
