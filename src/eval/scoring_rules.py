from __future__ import annotations

import math
import re
from typing import Any, Iterable
from urllib.parse import urlparse

from ..answer_schema import AnswerSection, ClaimItem
from ..domain_docs import DEFAULT_DOCS
from ..evidence import EvidenceItem, normalize_source_id
from .schemas import BenchmarkCase, CaseWeightOverride, ModelPricing, Pricing, ScoreWeights

_FAILURE_TEXT_PATTERNS = [
    r"agent execution failed",
    r"request timeout",
    r"unexpected error",
]
_HANGUL_PATTERN = re.compile(r"[가-힣]")
_COMPARISON_MARKERS = (
    "비교",
    "차이",
    "반면",
    "다르",
    "contrast",
    "compare",
    "difference",
    "however",
    "whereas",
)


def _contains_any_pattern(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def _normalize_domain(url_or_domain: str) -> str:
    parsed = urlparse(url_or_domain if "://" in url_or_domain else f"https://{url_or_domain}")
    domain = (parsed.netloc or parsed.path).strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


_ALLOWED_OFFICIAL_DOMAINS = {_normalize_domain(value) for value in DEFAULT_DOCS.values()}


def _is_valid_official_source(url_or_path: str) -> bool:
    parsed = urlparse(str(url_or_path or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return False
    return _normalize_domain(parsed.netloc) in _ALLOWED_OFFICIAL_DOMAINS


def _is_valid_local_source(url_or_path: str) -> bool:
    raw = str(url_or_path or "").strip()
    if not raw:
        return False
    normalized = raw.replace("\\", "/").lower()
    return (
        normalized.endswith(".py")
        or normalized.endswith(".ipynb")
        or "/uploads/" in normalized
        or normalized.startswith("uploads/")
        or normalized.startswith("data/")
    )


def _expected_local_citation_tool(case: BenchmarkCase) -> str:
    return "upload_search" if case.upload_fixture else "rag_search"


def _contains_hangul(text: str) -> bool:
    return bool(_HANGUL_PATTERN.search(str(text or "")))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _collect_valid_source_ids(
    evidence: list[EvidenceItem],
    *,
    required_kind: str,
    required_tool: str,
    source_validator: Any,
) -> set[str]:
    valid_ids: set[str] = set()
    for item in evidence:
        if item.kind != required_kind:
            continue
        if item.tool != required_tool:
            continue
        source_id = str(item.source_id or "").strip()
        document_id = str(item.document_id or normalize_source_id(item.url_or_path)).strip()
        if not source_id or not document_id:
            continue
        if document_id != normalize_source_id(item.url_or_path):
            continue
        if not source_validator(item.url_or_path):
            continue
        valid_ids.add(source_id)
    return valid_ids


def _copy_penalty(response_text: str, observed_evidence: list[EvidenceItem]) -> float:
    normalized_response = _normalize_text(response_text)
    if not normalized_response:
        return 0.0
    longest_match = 0
    for item in observed_evidence:
        snippet = _normalize_text(item.snippet or "")
        if len(snippet) < 48:
            continue
        if snippet and snippet in normalized_response:
            longest_match = max(longest_match, len(snippet))
    if longest_match >= 120:
        return 0.65
    if longest_match >= 72:
        return 0.45
    return 0.0


def _hybrid_comparison_present(
    response_text: str,
    response_sections: list[AnswerSection] | None = None,
) -> bool:
    if any(str(section.kind or "").strip() == "comparison" for section in (response_sections or [])):
        return True
    normalized = _normalize_text(response_text)
    return any(marker in normalized for marker in _COMPARISON_MARKERS)


def resolve_effective_weights(
    *,
    case: BenchmarkCase | None = None,
    base_weights: ScoreWeights,
    case_override: CaseWeightOverride | None,
) -> tuple[ScoreWeights, str | None]:
    if (
        case is not None
        and case.category == "tool_action"
        and not case.require_official_citation
        and not case.require_local_citation
    ):
        merged = {
            "answer_quality": 0.40,
            "groundedness": 0.025,
            "citation_traceability": 0.025,
            "tool_choice": 0.30,
            "format_language": 0.10,
            "llm_judge": 0.15,
        }
    else:
        merged = base_weights.as_dict()
    if case_override is not None:
        merged.update(case_override.as_partial_dict())

    for key, value in merged.items():
        if value < 0.0 or not math.isfinite(float(value)):
            return base_weights, f"invalid weight '{key}': {value}"

    total = float(sum(merged.values()))
    if total <= 0.0 or not math.isfinite(total):
        return base_weights, "weight sum must be a positive finite number"

    normalized = {key: float(value) / total for key, value in merged.items()}
    try:
        return ScoreWeights(**normalized), None
    except Exception as exc:
        return base_weights, f"failed to build normalized weights: {exc}"


def resolve_base_weights_for_case(
    *,
    case: BenchmarkCase,
    base_weights: ScoreWeights,
) -> ScoreWeights:
    if case.category != "tool_action":
        return base_weights
    return ScoreWeights(
        answer_quality=0.35,
        groundedness=0.10,
        citation_traceability=0.05,
        tool_choice=0.25,
        format_language=0.10,
        llm_judge=0.15,
    )


def score_tool_choice(case: BenchmarkCase, called_tools: list[str]) -> float:
    expected = set(case.expected_tools)
    forbidden = set(case.forbidden_tools)
    called = set(called_tools)

    if not expected and not forbidden:
        return 1.0

    expected_score = 1.0
    if expected:
        expected_score = len(expected.intersection(called)) / len(expected)

    forbidden_penalty = 0.0
    if forbidden:
        forbidden_penalty = len(forbidden.intersection(called)) / len(forbidden)

    return max(0.0, expected_score * (1.0 - forbidden_penalty))


def score_answer_quality(
    case: BenchmarkCase,
    response_text: str,
    observed_evidence: list[EvidenceItem],
    *,
    response_sections: list[AnswerSection] | None = None,
    synthesis_mode: str | None = None,
) -> float:
    text = response_text or ""
    if not text.strip():
        return 0.0

    include_score = 1.0
    if case.must_include:
        include_hits = sum(1 for needle in case.must_include if needle.lower() in text.lower())
        include_score = include_hits / len(case.must_include)

    exclude_score = 1.0
    if case.must_not_include:
        exclude_violations = sum(1 for needle in case.must_not_include if needle.lower() in text.lower())
        exclude_score = 1.0 - (exclude_violations / len(case.must_not_include))

    quality = max(0.0, min(1.0, (include_score + exclude_score) / 2.0))

    copy_penalty = _copy_penalty(text, observed_evidence)
    if copy_penalty > 0.0:
        quality = max(0.0, quality - copy_penalty)

    if case.category == "hybrid" and not _hybrid_comparison_present(text, response_sections):
        quality = min(quality, 0.25)

    if case.category in {"docs_only", "hybrid"} and synthesis_mode == "deterministic_grounded_direct":
        quality = min(quality, 0.2)

    return max(0.0, min(1.0, quality))


def score_citation_traceability(
    case: BenchmarkCase,
    response_evidence: list[EvidenceItem],
    observed_evidence: list[EvidenceItem],
    called_tools: list[str],
) -> float:
    checks: list[bool] = []

    if case.require_official_citation:
        response_ids = _collect_valid_source_ids(
            response_evidence,
            required_kind="official",
            required_tool="tavily_search",
            source_validator=_is_valid_official_source,
        )
        observed_ids = _collect_valid_source_ids(
            observed_evidence,
            required_kind="official",
            required_tool="tavily_search",
            source_validator=_is_valid_official_source,
        )
        checks.append(("tavily_search" in called_tools) and bool(response_ids.intersection(observed_ids)))

    if case.require_local_citation:
        expected_local_tool = _expected_local_citation_tool(case)
        response_ids = _collect_valid_source_ids(
            response_evidence,
            required_kind="local",
            required_tool=expected_local_tool,
            source_validator=_is_valid_local_source,
        )
        observed_ids = _collect_valid_source_ids(
            observed_evidence,
            required_kind="local",
            required_tool=expected_local_tool,
            source_validator=_is_valid_local_source,
        )
        checks.append((expected_local_tool in called_tools) and bool(response_ids.intersection(observed_ids)))

    if not checks:
        return 1.0

    return sum(1 for check in checks if check) / len(checks)


def _claim_support_ratio(
    response_claims: list[ClaimItem] | None,
    observed_evidence: list[EvidenceItem],
) -> float | None:
    if not response_claims:
        return None
    observed_ids = {
        str(item.source_id or item.document_id or "").strip()
        for item in observed_evidence
        if str(item.source_id or item.document_id or "").strip()
    }
    if not observed_ids:
        return 0.0
    supported = 0
    for claim in response_claims:
        if any(evidence_id in observed_ids for evidence_id in claim.evidence_ids):
            supported += 1
    return supported / len(response_claims)


def score_groundedness(
    *,
    case: BenchmarkCase | None = None,
    response_text: str,
    response_evidence: list[EvidenceItem],
    observed_evidence: list[EvidenceItem],
    validator_reason: str | None = None,
    response_claims: list[ClaimItem] | None = None,
    invalid_claim_count: int = 0,
    **_unused: Any,
) -> float:
    text = (response_text or "").strip()
    if not text:
        return 0.0
    if case is not None and case.category == "tool_action":
        if not response_evidence:
            if not observed_evidence:
                return 1.0
            if not case.require_official_citation and not case.require_local_citation:
                return 1.0
    if not observed_evidence or not response_evidence:
        return 0.0

    observed_ids = {
        str(item.source_id or item.document_id or "").strip()
        for item in observed_evidence
        if str(item.source_id or item.document_id or "").strip()
    }
    response_ids = {
        str(item.source_id or item.document_id or "").strip()
        for item in response_evidence
        if str(item.source_id or item.document_id or "").strip()
    }
    if not observed_ids or not response_ids:
        return 0.0

    score = len(response_ids.intersection(observed_ids)) / len(response_ids)
    claim_support_ratio = _claim_support_ratio(response_claims, observed_evidence)
    if claim_support_ratio is not None:
        score = min(score, claim_support_ratio)
    if invalid_claim_count > 0:
        score = min(score, max(0.0, 1.0 - (0.4 * invalid_claim_count)))

    if validator_reason == "no_evidence":
        return 0.0
    if validator_reason == "unsupported_claims":
        score = min(score, 0.2)
    elif validator_reason == "low_score":
        score = min(score, 0.35)
    elif validator_reason == "tool_error":
        score = min(score, 0.4)

    return max(0.0, min(1.0, score))


def score_format_language(
    *,
    case: BenchmarkCase,
    runtime_errors: list[str],
    response_errors: list[str],
    judge_errors: list[str],
    response_text: str,
) -> float:
    if runtime_errors or response_errors:
        return 0.0

    text = (response_text or "").strip()
    if not text:
        return 0.0
    if _contains_any_pattern(text, _FAILURE_TEXT_PATTERNS):
        return 0.0
    if _contains_hangul(case.query) and not _contains_hangul(text):
        return 0.0
    if any(str(error).startswith("invalid_eval:") for error in judge_errors):
        return 0.0
    return 1.0


def compute_rule_scores(
    *,
    case: BenchmarkCase,
    response_text: str,
    called_tools: list[str],
    response_evidence: list[EvidenceItem],
    observed_evidence: list[EvidenceItem],
    runtime_errors: list[str],
    response_errors: list[str],
    judge_errors: list[str],
    validator_reason: str | None = None,
    response_claims: list[ClaimItem] | None = None,
    retrieval_diagnostics: list[Any] | None = None,
    synthesis_mode: str | None = None,
    valid_claim_count: int = 0,
    invalid_claim_count: int = 0,
    response_sections: list[AnswerSection] | None = None,
    **_unused: Any,
) -> dict[str, float]:
    _ = retrieval_diagnostics, valid_claim_count
    return {
        "answer_quality": score_answer_quality(
            case,
            response_text,
            observed_evidence,
            response_sections=response_sections,
            synthesis_mode=synthesis_mode,
        ),
        "groundedness": score_groundedness(
            case=case,
            response_text=response_text,
            response_evidence=response_evidence,
            observed_evidence=observed_evidence,
            validator_reason=validator_reason,
            response_claims=response_claims,
            invalid_claim_count=invalid_claim_count,
        ),
        "citation_traceability": score_citation_traceability(
            case=case,
            response_evidence=response_evidence,
            observed_evidence=observed_evidence,
            called_tools=called_tools,
        ),
        "tool_choice": score_tool_choice(case, called_tools),
        "format_language": score_format_language(
            case=case,
            runtime_errors=runtime_errors,
            response_errors=response_errors,
            judge_errors=judge_errors,
            response_text=response_text,
        ),
    }


def compute_rule_weighted_score(
    component_scores: dict[str, float],
    weights: ScoreWeights,
) -> float:
    weight_map = weights.as_dict()
    score = 0.0
    for key, value in component_scores.items():
        score += value * float(weight_map.get(key, 0.0))
    return max(0.0, min(1.0, score))


def compute_composite_quality_score(
    rule_weighted_score: float,
    llm_judge_score: float | None,
    weights: ScoreWeights,
) -> float:
    llm_weight = float(weights.llm_judge)
    if llm_judge_score is None:
        denominator = max(1e-9, 1.0 - llm_weight)
        normalized = rule_weighted_score / denominator
        return max(0.0, min(1.0, normalized))
    return max(0.0, min(1.0, rule_weighted_score + llm_judge_score * llm_weight))


def compute_final_score(
    rule_weighted_score: float,
    llm_judge_score: float | None,
    weights: ScoreWeights,
) -> float:
    return compute_composite_quality_score(
        rule_weighted_score=rule_weighted_score,
        llm_judge_score=llm_judge_score,
        weights=weights,
    )


def score_tool_match(case: BenchmarkCase, called_tools: list[str]) -> float:
    return score_tool_choice(case, called_tools)


def score_content_constraints(case: BenchmarkCase, response_text: str) -> float:
    return score_answer_quality(case, response_text, [])


def score_citation_compliance(
    case: BenchmarkCase,
    response_evidence: list[EvidenceItem],
    observed_evidence: list[EvidenceItem],
    called_tools: list[str],
) -> float:
    return score_citation_traceability(
        case=case,
        response_evidence=response_evidence,
        observed_evidence=observed_evidence,
        called_tools=called_tools,
    )


def score_safety_format(
    *,
    runtime_errors: list[str],
    response_errors: list[str],
    judge_errors: list[str],
    response_text: str,
) -> float:
    return score_format_language(
        case=BenchmarkCase(case_id="legacy", category="tool_action", query=""),
        runtime_errors=runtime_errors,
        response_errors=response_errors,
        judge_errors=judge_errors,
        response_text=response_text,
    )


def _resolve_model_pricing(model_name: str | None, pricing: Pricing) -> ModelPricing:
    if model_name:
        configured_pricing = pricing.models.get(str(model_name))
        if configured_pricing is not None:
            return configured_pricing
    return ModelPricing(
        prompt_per_1k_usd=float(pricing.prompt_per_1k_usd),
        completion_per_1k_usd=float(pricing.completion_per_1k_usd),
    )


def _extract_usage_from_llm_call(llm_call: Any) -> tuple[int, int]:
    if not isinstance(llm_call, dict):
        return 0, 0

    usage_metadata = llm_call.get("usage_metadata")
    response_metadata = llm_call.get("response_metadata")
    usage_candidates = []
    if isinstance(usage_metadata, dict):
        usage_candidates.append(usage_metadata)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            usage_candidates.append(token_usage)

    for usage in usage_candidates:
        try:
            prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            completion_tokens = int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )
        except (TypeError, ValueError):
            continue
        return max(0, prompt_tokens), max(0, completion_tokens)

    return 0, 0


def _extract_model_name_from_llm_call(llm_call: Any) -> str | None:
    if not isinstance(llm_call, dict):
        return None
    response_metadata = llm_call.get("response_metadata")
    if not isinstance(response_metadata, dict):
        return None
    model_name = response_metadata.get("model_name") or response_metadata.get("model")
    return str(model_name) if model_name else None


def compute_cost_usd(
    *,
    token_usage: Any,
    llm_calls: list[Any] | None = None,
    pricing: Pricing,
) -> float | None:
    if llm_calls:
        total_cost = 0.0
        has_usage = False
        for call in llm_calls:
            prompt_tokens, completion_tokens = _extract_usage_from_llm_call(call)
            if prompt_tokens <= 0 and completion_tokens <= 0:
                continue
            has_usage = True
            model_pricing = _resolve_model_pricing(_extract_model_name_from_llm_call(call), pricing)
            total_cost += (prompt_tokens / 1000.0) * float(model_pricing.prompt_per_1k_usd)
            total_cost += (completion_tokens / 1000.0) * float(model_pricing.completion_per_1k_usd)
        if has_usage:
            return round(total_cost, 8)

    if token_usage is None:
        return None
    prompt_tokens = int(getattr(token_usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(token_usage, "completion_tokens", 0) or 0)
    prompt_cost = (prompt_tokens / 1000.0) * float(pricing.prompt_per_1k_usd)
    completion_cost = (completion_tokens / 1000.0) * float(pricing.completion_per_1k_usd)
    return round(prompt_cost + completion_cost, 8)


def tool_confusion_counts(case: BenchmarkCase, called_tools: list[str]) -> tuple[int, int, int]:
    expected = set(case.expected_tools)
    forbidden = set(case.forbidden_tools)
    called = set(called_tools)

    tp = len(expected.intersection(called))
    fn = len(expected.difference(called))
    fp = 0
    if forbidden:
        fp += len(forbidden.intersection(called))
    if expected:
        fp += len(called.difference(expected).difference(forbidden))
    return tp, fp, fn
