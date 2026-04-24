from __future__ import annotations

from dataclasses import dataclass

from src.core.planner_schema import PlannerOutput
from src.core.prompts import needs_save, needs_slack


@dataclass(frozen=True, slots=True)
class SynthesisBudgetProfile:
    category: str
    max_tokens: int
    snippet_chars: int
    evidence_chars: int
    max_evidence_items: int
    max_claims: int | None = None
    max_section_sentences: int | None = None


_TOOL_ACTION_PROFILE = SynthesisBudgetProfile(
    category="tool_action",
    max_tokens=512,
    snippet_chars=180,
    evidence_chars=500,
    max_evidence_items=1,
)
_STANDARD_PROFILE_BY_CATEGORY: dict[str, SynthesisBudgetProfile] = {
    "docs_only": SynthesisBudgetProfile(
        category="docs_only",
        max_tokens=1024,
        snippet_chars=300,
        evidence_chars=1200,
        max_evidence_items=1,
    ),
    "rag_only": SynthesisBudgetProfile(
        category="rag_only",
        max_tokens=1024,
        snippet_chars=300,
        evidence_chars=1200,
        max_evidence_items=1,
    ),
    "upload_only": SynthesisBudgetProfile(
        category="upload_only",
        max_tokens=1024,
        snippet_chars=300,
        evidence_chars=1200,
        max_evidence_items=1,
    ),
}
_HYBRID_PROFILE = SynthesisBudgetProfile(
    category="hybrid",
    max_tokens=1280,
    snippet_chars=260,
    evidence_chars=1100,
    max_evidence_items=2,
    max_claims=4,
    max_section_sentences=3,
)
_GENERAL_PROFILE = SynthesisBudgetProfile(
    category="general",
    max_tokens=1024,
    snippet_chars=300,
    evidence_chars=1200,
    max_evidence_items=1,
)
_EXPANDED_SECTION_MARKERS = (
    "summary",
    "checklist",
    "step by step",
    "요약",
    "체크리스트",
    "단계",
    "순서",
)


def _requested_routes(planner_output: PlannerOutput) -> list[str]:
    if not planner_output.use_retrieval:
        return []
    routes: list[str] = []
    for task in planner_output.tasks or []:
        route = str(task.route or "").strip()
        if route and route not in routes:
            routes.append(route)
    return routes


def _allows_two_standard_evidence_items(user_input: str) -> bool:
    lowered = str(user_input or "").lower()
    return any(marker in lowered for marker in _EXPANDED_SECTION_MARKERS)


def resolve_synthesis_budget_profile(
    *,
    user_input: str,
    planner_output: PlannerOutput,
    synthesis_max_tokens: int,
) -> SynthesisBudgetProfile:
    if needs_save(user_input) or needs_slack(user_input):
        base_profile = _TOOL_ACTION_PROFILE
    else:
        routes = _requested_routes(planner_output)
        route_set = set(routes)
        if "docs" in route_set and route_set.intersection({"upload", "local"}):
            base_profile = _HYBRID_PROFILE
        elif route_set == {"docs"}:
            base_profile = _STANDARD_PROFILE_BY_CATEGORY["docs_only"]
        elif route_set == {"local"}:
            base_profile = _STANDARD_PROFILE_BY_CATEGORY["rag_only"]
        elif route_set == {"upload"}:
            base_profile = _STANDARD_PROFILE_BY_CATEGORY["upload_only"]
        else:
            base_profile = _GENERAL_PROFILE

    max_evidence_items = base_profile.max_evidence_items
    if (
        base_profile.category in {"docs_only", "rag_only", "upload_only"}
        and _allows_two_standard_evidence_items(user_input)
    ):
        max_evidence_items = 2

    return SynthesisBudgetProfile(
        category=base_profile.category,
        max_tokens=max(1, min(int(synthesis_max_tokens), int(base_profile.max_tokens))),
        snippet_chars=base_profile.snippet_chars,
        evidence_chars=base_profile.evidence_chars,
        max_evidence_items=max_evidence_items,
        max_claims=base_profile.max_claims,
        max_section_sentences=base_profile.max_section_sentences,
    )


def compact_synthesis_budget_profile(profile: SynthesisBudgetProfile) -> SynthesisBudgetProfile:
    return SynthesisBudgetProfile(
        category=profile.category,
        max_tokens=max(1, int(profile.max_tokens) // 2),
        snippet_chars=max(80, int(profile.snippet_chars) // 2),
        evidence_chars=max(350, int(profile.evidence_chars) // 2),
        max_evidence_items=profile.max_evidence_items,
        max_claims=profile.max_claims,
        max_section_sentences=profile.max_section_sentences,
    )


__all__ = [
    "SynthesisBudgetProfile",
    "compact_synthesis_budget_profile",
    "resolve_synthesis_budget_profile",
]
