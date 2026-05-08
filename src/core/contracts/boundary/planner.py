from __future__ import annotations

from typing import Any

from src.core.planner_schema import PlannerOutput, normalize_planner_output_input
from src.core.contracts.debug import PlannerDiagnostic, empty_planner_diagnostic
from src.core.contracts.graph_state import PlannerState
from src.core.contracts.routes import normalize_routes


def _coerce_planner_payload(raw: Any) -> Any:
    if isinstance(raw, PlannerOutput):
        return raw
    return normalize_planner_output_input(raw)


def parse_planner_output(raw: Any, errors: list[str], warnings: list[str] | None = None) -> PlannerOutput:
    payload = _coerce_planner_payload(raw)
    if isinstance(payload, PlannerOutput):
        return payload
    try:
        return PlannerOutput.validate_input(payload, warnings=warnings)
    except Exception as exc:
        errors.append(f"planner: output validation failed ({exc})")
        return PlannerOutput.fallback()


def parse_planner_diagnostic(value: Any) -> PlannerDiagnostic | None:
    if value is None:
        return None
    if isinstance(value, PlannerDiagnostic):
        return value
    if not isinstance(value, dict):
        return None

    fallback_routes = value.get("fallback_routes")
    required_routes = value.get("required_routes")
    status = value.get("status")
    reason = value.get("reason")
    override_reason = value.get("override_reason")
    planner_warnings = value.get("planner_warnings")
    if override_reason not in {
        "missing_required_retrieval",
        "missing_required_routes",
        "upload_retriever_missing",
    }:
        override_reason = None

    return PlannerDiagnostic(
        status=str(status) if status is not None else "",
        reason=str(reason) if reason is not None else None,
        fallback_routes=normalize_routes(fallback_routes) if isinstance(fallback_routes, list) else [],
        intent_required=bool(value.get("intent_required", False)),
        required_routes=normalize_routes(required_routes) if isinstance(required_routes, list) else [],
        override_applied=bool(value.get("override_applied", False)),
        override_reason=override_reason,
        planner_warnings=[
            str(warning).strip()
            for warning in planner_warnings
            if str(warning).strip()
        ]
        if isinstance(planner_warnings, list)
        else [],
    )


def parse_planner_state(value: Any) -> PlannerState:
    if isinstance(value, PlannerState):
        return value
    if not isinstance(value, dict):
        return PlannerState()

    planner_errors: list[str] = []
    planner_warnings: list[str] = []
    diagnostics = value.get("diagnostics")
    status = value.get("status")
    if status not in {"llm", "deterministic", "heuristic_fallback", "fallback_no_routes"}:
        status = "llm"
    planner_diagnostics = parse_planner_diagnostic(diagnostics)
    output = parse_planner_output(value.get("output"), planner_errors, planner_warnings)
    if planner_warnings:
        merged_warnings = list(planner_diagnostics.planner_warnings) if planner_diagnostics else []
        for warning in planner_warnings:
            if warning not in merged_warnings:
                merged_warnings.append(warning)
        base_diagnostics = planner_diagnostics or empty_planner_diagnostic(status=status)
        planner_diagnostics = base_diagnostics.model_copy(
            update={"planner_warnings": merged_warnings}
        )
    return PlannerState(
        output=output,
        status=status,
        diagnostics=planner_diagnostics or empty_planner_diagnostic(status=status),
        guided_followup=(
            str(value.get("guided_followup")).strip()
            if value.get("guided_followup") is not None
            else None
        ),
    )


def get_planner_state(state: dict[str, Any]) -> PlannerState:
    return parse_planner_state(state.get("planner"))
