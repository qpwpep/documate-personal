from __future__ import annotations

from ..result_models import CaseResult


def result_server_latency_ms(result: CaseResult) -> int | None:
    if result.latency_ms_server is not None:
        return result.latency_ms_server
    if result.latency_breakdown is not None:
        return result.latency_breakdown.server_total_ms
    return None


def _stage_total_field_for_stage(stage_name: str) -> str:
    if stage_name in {"retrieval", "synthesis"}:
        return f"{stage_name}_total_ms"
    return f"{stage_name}_ms"


def result_latency_breakdown_stage_ms(result: CaseResult, stage_field: str) -> int | None:
    breakdown = result.latency_breakdown
    if breakdown is None:
        return None
    if stage_field == "upload_retriever_build_ms":
        return breakdown.upload_retriever_build_ms
    stage_totals = breakdown.stage_totals_ms
    if stage_field in stage_totals.model_fields_set:
        return int(getattr(stage_totals, stage_field, 0) or 0)
    attempt_values = [
        int(attempt.latency_ms)
        for attempt in breakdown.stage_attempts
        if _stage_total_field_for_stage(str(attempt.stage)) == stage_field
    ]
    if attempt_values:
        return sum(attempt_values)
    return None
