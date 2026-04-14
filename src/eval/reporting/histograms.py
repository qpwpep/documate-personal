from __future__ import annotations

from collections import Counter, defaultdict

from ...contracts.routes import ROUTE_ORDER, route_for_tool, sort_routes
from ..schemas import (
    AnalysisStats,
    BenchmarkCase,
    CaseResult,
    CategoryPassRate,
    LatencyBreakdownCoverage,
    PlannerDiagnosticsBucket,
    PlannerErrorBucket,
    RetrievalRouteStatusBucket,
    RetrievalWarningBucket,
    RouteConfusionBucket,
    StageLatencyPercentile,
    SynthesisModeBucket,
    ValidatorReasonBucket,
)


_CATEGORY_ORDER: tuple[str, ...] = ("docs_only", "rag_only", "hybrid", "tool_action")
_PLANNER_ERROR_ORDER: tuple[str, ...] = (
    "structured_output_invocation_failed",
    "output_validation_failed",
    "sanitized_output_validation_failed",
    "upload_route_dropped",
)
_LATENCY_STAGE_FIELDS: tuple[str, ...] = (
    "upload_retriever_build_ms",
    "summarize_ms",
    "planner_ms",
    "retrieval_total_ms",
    "synthesis_total_ms",
    "validation_ms",
    "action_postprocess_ms",
)


def percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = rank - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac)


def _category_sort_key(value: str) -> tuple[int, str]:
    if value in _CATEGORY_ORDER:
        return (_CATEGORY_ORDER.index(value), value)
    return (len(_CATEGORY_ORDER), value)


def _route_sort_key(value: str) -> tuple[int, str]:
    if value in ROUTE_ORDER:
        return (ROUTE_ORDER.index(value), value)
    return (len(ROUTE_ORDER), value)


def _planner_error_sort_key(value: str) -> tuple[int, str]:
    if value in _PLANNER_ERROR_ORDER:
        return (_PLANNER_ERROR_ORDER.index(value), value)
    return (len(_PLANNER_ERROR_ORDER), value)


def _sort_categories(values: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=_category_sort_key)


def _sort_routes(values: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    return sort_routes(values)


def _normalize_reason_text(text: str | None, *, max_length: int = 160) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return ""
    if len(normalized) <= max_length:
        return normalized
    truncated = normalized[: max_length - 3].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.rstrip(" .,:;") + "..."


def _build_rule_score_signature(result: CaseResult) -> str:
    low_scores = [
        (str(name), float(value))
        for name, value in result.rule_scores.items()
        if float(value) < 1.0
    ]
    if result.judge_subscores is not None:
        low_scores.extend(
            [
                (f"judge.{name}", float(value))
                for name, value in result.judge_subscores.model_dump().items()
                if float(value) < 1.0
            ]
        )
    if low_scores:
        low_scores.sort(key=lambda item: (item[1], item[0]))
        signature = ", ".join(f"{name}={value:.2f}" for name, value in low_scores[:3])
        return f"low scores: {signature}"
    if result.composite_quality_score is not None:
        return f"composite_quality_score={float(result.composite_quality_score):.2f} below threshold"
    return "score below threshold"


def _result_retrieval_warnings(result: CaseResult) -> list[str]:
    return sorted(
        {
            str(warning).strip()
            for diagnostic in result.retrieval_diagnostics
            for warning in diagnostic.warnings
            if str(warning).strip()
        }
    )


def build_failure_reason(result: CaseResult) -> str:
    if result.runtime_errors:
        return ", ".join(result.runtime_errors)
    if result.response_errors:
        return ", ".join(result.response_errors)
    if result.judge_errors:
        return ", ".join(result.judge_errors)
    warnings = _result_retrieval_warnings(result)
    if warnings:
        return "retrieval_warning:" + ", ".join(warnings)
    if result.validator_reason:
        return f"validator:{result.validator_reason}"
    if result.llm_judge_reason:
        return _normalize_reason_text(result.llm_judge_reason)
    if result.gate_failures:
        return ", ".join(result.gate_failures)
    return _build_rule_score_signature(result)


def _tool_names_to_routes(tool_names: list[str]) -> list[str]:
    return _sort_routes({route_for_tool(tool_name) for tool_name in tool_names if route_for_tool(tool_name)})


def _observed_routes(result: CaseResult) -> list[str]:
    diagnostic_routes = _sort_routes(
        {
            str(item.route or route_for_tool(item.tool or "")).strip()
            for item in result.retrieval_diagnostics
            if str(item.route or route_for_tool(item.tool or "")).strip()
        }
    )
    if diagnostic_routes:
        return diagnostic_routes
    return _tool_names_to_routes(result.tool_calls)


def _build_category_pass_rates(results: list[CaseResult]) -> list[CategoryPassRate]:
    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    for result in results:
        per_category[result.category]["total"] += 1
        per_category[result.category]["passed"] += 1 if bool(result.release_pass) else 0

    rows: list[CategoryPassRate] = []
    for category in _sort_categories(list(per_category.keys())):
        counts = per_category[category]
        total_cases = counts["total"]
        pass_rate = (counts["passed"] / total_cases) if total_cases else 0.0
        rows.append(
            CategoryPassRate(
                category=category,
                passed_cases=counts["passed"],
                total_cases=total_cases,
                pass_rate=round(pass_rate, 4),
            )
        )
    return rows


def _build_planner_diagnostics_histogram(results: list[CaseResult]) -> list[PlannerDiagnosticsBucket]:
    counter: Counter[tuple[str, str, str | None, str | None]] = Counter()
    for result in results:
        if result.planner_diagnostics is None:
            counter[(result.category, "missing", "diagnostics_unavailable", None)] += 1
            continue
        counter[
            (
                result.category,
                str(result.planner_diagnostics.status or "missing"),
                result.planner_diagnostics.reason,
                result.planner_diagnostics.override_reason,
            )
        ] += 1

    rows = [
        PlannerDiagnosticsBucket(
            category=category,
            status=status,
            reason=reason,
            override_reason=override_reason,
            count=count,
        )
        for (category, status, reason, override_reason), count in counter.items()
    ]
    rows.sort(key=lambda item: (_category_sort_key(item.category), -item.count, item.status, item.reason or "", item.override_reason or ""))
    return rows


def _normalize_planner_error_code(error: str) -> str | None:
    normalized = str(error or "").strip().lower()
    if not normalized:
        return None
    if "structured_output_invocation_failed" in normalized or "structured output invocation failed" in normalized:
        return "structured_output_invocation_failed"
    if "sanitized_output_validation_failed" in normalized or "sanitized output validation failed" in normalized:
        return "sanitized_output_validation_failed"
    if "upload_route_dropped" in normalized or "dropped upload route because retriever is unavailable" in normalized:
        return "upload_route_dropped"
    if "output_validation_failed" in normalized or "output validation failed" in normalized:
        return "output_validation_failed"
    return None


def _build_planner_error_histogram(results: list[CaseResult]) -> list[PlannerErrorBucket]:
    counter: Counter[tuple[str, str]] = Counter()
    for result in results:
        for error in result.planner_errors:
            code = _normalize_planner_error_code(error)
            if code is not None:
                counter[(result.category, code)] += 1
    rows = [PlannerErrorBucket(category=category, error_code=error_code, count=count) for (category, error_code), count in counter.items()]
    rows.sort(key=lambda item: (_category_sort_key(item.category), _planner_error_sort_key(item.error_code), -item.count))
    return rows


def _build_retrieval_route_status_histogram(results: list[CaseResult]) -> list[RetrievalRouteStatusBucket]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for result in results:
        for diagnostic in result.retrieval_diagnostics:
            route = str(diagnostic.route or route_for_tool(diagnostic.tool or "")).strip()
            if not route:
                continue
            status = str(diagnostic.status or "unknown").strip() or "unknown"
            counter[(result.category, route, status)] += 1
    rows = [RetrievalRouteStatusBucket(category=category, route=route, status=status, count=count) for (category, route, status), count in counter.items()]
    rows.sort(key=lambda item: (_category_sort_key(item.category), _route_sort_key(item.route), -item.count, item.status))
    return rows


def _build_retrieval_warning_histogram(results: list[CaseResult]) -> list[RetrievalWarningBucket]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for result in results:
        for diagnostic in result.retrieval_diagnostics:
            route = str(diagnostic.route or route_for_tool(diagnostic.tool or "")).strip()
            if not route:
                continue
            for warning in diagnostic.warnings:
                normalized_warning = str(warning or "").strip()
                if normalized_warning:
                    counter[(result.category, route, normalized_warning)] += 1
    rows = [
        RetrievalWarningBucket(category=category, route=route, warning=warning, count=count)
        for (category, route, warning), count in counter.items()
    ]
    rows.sort(key=lambda item: (_category_sort_key(item.category), _route_sort_key(item.route), -item.count, item.warning))
    return rows


def _build_route_confusion(
    *,
    case_map: dict[str, BenchmarkCase],
    results: list[CaseResult],
) -> list[RouteConfusionBucket]:
    counter: Counter[
        tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    ] = Counter()
    for result in results:
        case = case_map.get(result.case_id)
        if case is None:
            continue
        expected_routes = set(_tool_names_to_routes(case.expected_tools))
        forbidden_routes = set(_tool_names_to_routes(case.forbidden_tools))
        observed_routes = set(_observed_routes(result))
        missing_expected = expected_routes.difference(observed_routes)
        forbidden_observed = observed_routes.intersection(forbidden_routes)
        unexpected_routes = observed_routes.difference(expected_routes).difference(forbidden_routes)
        if not (missing_expected or forbidden_observed or unexpected_routes):
            continue
        counter[
            (
                result.category,
                tuple(_sort_routes(expected_routes)),
                tuple(_sort_routes(observed_routes)),
                tuple(_sort_routes(missing_expected)),
                tuple(_sort_routes(unexpected_routes)),
                tuple(_sort_routes(forbidden_observed)),
            )
        ] += 1

    rows = [
        RouteConfusionBucket(
            category=category,
            expected_routes=list(expected_routes),
            observed_routes=list(observed_routes),
            missing_expected_routes=list(missing_expected_routes),
            unexpected_routes=list(unexpected_routes),
            forbidden_routes=list(forbidden_routes),
            count=count,
        )
        for (
            category,
            expected_routes,
            observed_routes,
            missing_expected_routes,
            unexpected_routes,
            forbidden_routes,
        ), count in counter.items()
    ]
    rows.sort(key=lambda item: (_category_sort_key(item.category), -item.count, ",".join(item.expected_routes), ",".join(item.observed_routes)))
    return rows


def _build_validator_reason_histogram(results: list[CaseResult]) -> list[ValidatorReasonBucket]:
    counter: Counter[tuple[str, str]] = Counter()
    totals_by_category: Counter[str] = Counter()
    for result in results:
        if result.release_pass:
            continue
        reason = str(result.validator_reason or "missing")
        counter[(result.category, reason)] += 1
        totals_by_category[result.category] += 1
    rows: list[ValidatorReasonBucket] = []
    for (category, reason), count in counter.items():
        total_failed = int(totals_by_category.get(category, 0))
        share = (count / total_failed) if total_failed else 0.0
        rows.append(ValidatorReasonBucket(category=category, reason=reason, count=count, share=round(share, 4)))
    rows.sort(key=lambda item: (_category_sort_key(item.category), -item.count, item.reason))
    return rows


def _build_synthesis_mode_histogram(results: list[CaseResult]) -> list[SynthesisModeBucket]:
    counter: Counter[tuple[str, str]] = Counter()
    for result in results:
        if result.synthesis_mode:
            counter[(result.category, result.synthesis_mode)] += 1
            continue
        breakdown = result.latency_breakdown
        if breakdown is None or not breakdown.synthesis_attempts:
            continue
        counter[(result.category, breakdown.synthesis_attempts[0].mode)] += 1
    rows = [SynthesisModeBucket(category=category, mode=mode, count=count) for (category, mode), count in counter.items()]
    rows.sort(key=lambda item: (_category_sort_key(item.category), -item.count, item.mode))
    return rows


def _extract_latency_stage_value(result: CaseResult, stage_name: str) -> int | None:
    breakdown = result.latency_breakdown
    if breakdown is None:
        return None
    if stage_name == "upload_retriever_build_ms":
        return breakdown.upload_retriever_build_ms
    return int(getattr(breakdown.stage_totals_ms, stage_name, 0) or 0)


def _build_stage_latency_analysis(
    results: list[CaseResult],
) -> tuple[list[StageLatencyPercentile], LatencyBreakdownCoverage]:
    available_results = [result for result in results if result.latency_breakdown is not None]
    total_cases = len(results)
    available_cases = len(available_results)
    coverage = LatencyBreakdownCoverage(
        available_cases=available_cases,
        total_cases=total_cases,
        coverage_rate=round((available_cases / total_cases), 4) if total_cases else 0.0,
    )
    rows: list[StageLatencyPercentile] = []
    for stage_name in _LATENCY_STAGE_FIELDS:
        values = [
            value
            for result in available_results
            for value in [_extract_latency_stage_value(result, stage_name)]
            if value is not None
        ]
        if not values:
            continue
        p50_value = percentile(values, 0.50)
        p95_value = percentile(values, 0.95)
        rows.append(
            StageLatencyPercentile(
                stage=stage_name,
                sample_count=len(values),
                p50_latency_ms=round(p50_value, 2) if p50_value is not None else None,
                p95_latency_ms=round(p95_value, 2) if p95_value is not None else None,
            )
        )
    return rows, coverage


def build_analysis(*, case_map: dict[str, BenchmarkCase], results: list[CaseResult]) -> AnalysisStats:
    stage_latency_percentiles, latency_breakdown_coverage = _build_stage_latency_analysis(results)
    return AnalysisStats(
        category_pass_rates=_build_category_pass_rates(results),
        planner_diagnostics_histogram=_build_planner_diagnostics_histogram(results),
        planner_error_histogram=_build_planner_error_histogram(results),
        retrieval_route_status_histogram=_build_retrieval_route_status_histogram(results),
        retrieval_warning_histogram=_build_retrieval_warning_histogram(results),
        route_confusion=_build_route_confusion(case_map=case_map, results=results),
        validator_reason_histogram=_build_validator_reason_histogram(results),
        synthesis_mode_histogram=_build_synthesis_mode_histogram(results),
        stage_latency_percentiles=stage_latency_percentiles,
        latency_breakdown_coverage=latency_breakdown_coverage,
    )


__all__ = [
    "build_analysis",
    "build_failure_reason",
    "percentile",
]
