from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


StageName = Literal[
    "summarize",
    "planner",
    "retrieval",
    "synthesis",
    "validation",
    "action_postprocess",
]

SynthesisMode = Literal[
    "structured_only",
    "timeout_grounded_fallback",
    "structured_error_plain_fallback",
]

LatencyEventKind = Literal["stage", "retrieval_route", "synthesis_attempt"]

_STAGE_TOTAL_FIELD_BY_NAME: dict[StageName, str] = {
    "summarize": "summarize_ms",
    "planner": "planner_ms",
    "retrieval": "retrieval_total_ms",
    "synthesis": "synthesis_total_ms",
    "validation": "validation_ms",
    "action_postprocess": "action_postprocess_ms",
}


class StageTotalsModel(BaseModel):
    summarize_ms: int = 0
    planner_ms: int = 0
    retrieval_total_ms: int = 0
    synthesis_total_ms: int = 0
    validation_ms: int = 0
    action_postprocess_ms: int = 0


class StageAttemptModel(BaseModel):
    stage: StageName
    attempt: int = 1
    latency_ms: int = 0
    status: str | None = None


class RetrievalRouteLatencyModel(BaseModel):
    route: str = ""
    tool: str = ""
    attempt: int = 1
    latency_ms: int = 0
    status: str = ""


class SynthesisAttemptLatencyModel(BaseModel):
    attempt: int = 1
    mode: SynthesisMode
    structured_ms: int | None = None
    fallback_ms: int | None = None
    total_ms: int = 0


class LatencyBreakdownModel(BaseModel):
    server_total_ms: int | None = None
    graph_total_ms: int | None = None
    upload_retriever_build_ms: int | None = None
    stage_totals_ms: StageTotalsModel = Field(default_factory=StageTotalsModel)
    stage_attempts: list[StageAttemptModel] = Field(default_factory=list)
    retrieval_routes: list[RetrievalRouteLatencyModel] = Field(default_factory=list)
    synthesis_attempts: list[SynthesisAttemptLatencyModel] = Field(default_factory=list)


def elapsed_ms(start_perf: float, end_perf: float) -> int:
    return max(0, int(round((end_perf - start_perf) * 1000)))


def make_stage_latency_event(
    *,
    stage: StageName,
    attempt: int,
    latency_ms: int,
    status: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "stage",
        "stage": stage,
        "attempt": max(1, int(attempt)),
        "latency_ms": max(0, int(latency_ms)),
        "status": status,
    }


def make_retrieval_route_latency_event(
    *,
    route: str,
    tool: str,
    attempt: int,
    latency_ms: int,
    status: str,
) -> dict[str, Any]:
    return {
        "kind": "retrieval_route",
        "route": str(route or "").strip(),
        "tool": str(tool or "").strip(),
        "attempt": max(1, int(attempt)),
        "latency_ms": max(0, int(latency_ms)),
        "status": str(status or "").strip(),
    }


def make_synthesis_attempt_latency_event(
    *,
    attempt: int,
    mode: SynthesisMode,
    structured_ms: int | None,
    fallback_ms: int | None,
    total_ms: int,
) -> dict[str, Any]:
    return {
        "kind": "synthesis_attempt",
        "attempt": max(1, int(attempt)),
        "mode": mode,
        "structured_ms": None if structured_ms is None else max(0, int(structured_ms)),
        "fallback_ms": None if fallback_ms is None else max(0, int(fallback_ms)),
        "total_ms": max(0, int(total_ms)),
    }


def _coerce_trace_items(raw_trace: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_trace, list):
        return []
    return [item for item in raw_trace if isinstance(item, dict)]


def build_latency_breakdown(
    *,
    raw_trace: Any,
    graph_total_ms: int | None = None,
    server_total_ms: int | None = None,
    upload_retriever_build_ms: int | None = None,
) -> LatencyBreakdownModel:
    stage_totals = StageTotalsModel()
    stage_attempts: list[StageAttemptModel] = []
    retrieval_routes: list[RetrievalRouteLatencyModel] = []
    synthesis_attempts: list[SynthesisAttemptLatencyModel] = []

    for item in _coerce_trace_items(raw_trace):
        kind = str(item.get("kind") or "").strip()
        if kind == "stage":
            try:
                stage_attempt = StageAttemptModel.model_validate(item)
            except Exception:
                continue
            stage_attempts.append(stage_attempt)
            field_name = _STAGE_TOTAL_FIELD_BY_NAME.get(stage_attempt.stage)
            if field_name:
                current_value = int(getattr(stage_totals, field_name, 0) or 0)
                setattr(stage_totals, field_name, current_value + int(stage_attempt.latency_ms))
            continue
        if kind == "retrieval_route":
            try:
                retrieval_routes.append(RetrievalRouteLatencyModel.model_validate(item))
            except Exception:
                continue
            continue
        if kind == "synthesis_attempt":
            try:
                synthesis_attempts.append(SynthesisAttemptLatencyModel.model_validate(item))
            except Exception:
                continue

    return LatencyBreakdownModel(
        server_total_ms=server_total_ms,
        graph_total_ms=graph_total_ms,
        upload_retriever_build_ms=upload_retriever_build_ms,
        stage_totals_ms=stage_totals,
        stage_attempts=stage_attempts,
        retrieval_routes=retrieval_routes,
        synthesis_attempts=synthesis_attempts,
    )


def largest_latency_stage(breakdown: LatencyBreakdownModel | None) -> tuple[str | None, int | None]:
    if breakdown is None:
        return None, None

    candidates = {
        "upload_retriever_build_ms": breakdown.upload_retriever_build_ms,
        "summarize_ms": breakdown.stage_totals_ms.summarize_ms,
        "planner_ms": breakdown.stage_totals_ms.planner_ms,
        "retrieval_total_ms": breakdown.stage_totals_ms.retrieval_total_ms,
        "synthesis_total_ms": breakdown.stage_totals_ms.synthesis_total_ms,
        "validation_ms": breakdown.stage_totals_ms.validation_ms,
        "action_postprocess_ms": breakdown.stage_totals_ms.action_postprocess_ms,
    }

    valid_items = [
        (name, int(value))
        for name, value in candidates.items()
        if value is not None and int(value) > 0
    ]
    if not valid_items:
        return None, None

    stage_name, latency_ms = max(valid_items, key=lambda item: item[1])
    return stage_name, latency_ms
