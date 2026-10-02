from __future__ import annotations

from typing import Any

from src.core.answer_schema import AnswerResponse
from src.core.contracts import GraphState, ResponseState
from src.core.contracts.provenance import AnswerSource, BodyKind
from src.core.evidence import EvidenceRef
from src.core.llm_errors import ExecutionProblem, LLMDiagnostic


def build_synthesis_updates(
    *, debug: Any, result: AnswerResponse, evidence_packet: list[EvidenceRef], attempt: int,
    latency_trace: list[dict[str, Any]], retrieval_errors: list[str] | None = None,
    planner_errors: list[str] | None = None, synthesis_errors: list[str] | None = None,
    evidence_requirement_map: dict[str, list[str]] | None = None,
    kind: str = "draft",
    request_id: str | None = None,
    contract_revision: int = 0,
    normal_evidence_missing_requirement_ids: list[str] | None = None,
    body_kind: BodyKind | None = None,
    evidence_source: AnswerSource | None = None,
    problem: ExecutionProblem | None = None,
    llm_diagnostics: list[LLMDiagnostic] | None = None,
) -> GraphState:
    return {
        "response": ResponseState(result=result, evidence_packet=evidence_packet, synthesis_attempt=attempt,
                                  evidence_requirement_map=evidence_requirement_map or {}, kind=kind,
                                  normal_evidence_missing_requirement_ids=normal_evidence_missing_requirement_ids,
                                  request_id=request_id, contract_revision=contract_revision,
                                  body_kind=body_kind, evidence_source=evidence_source, problem=problem),
        "debug": debug.model_copy(update={
            "retrieval_errors": [*debug.retrieval_errors, *(retrieval_errors or [])],
            "planner_errors": [*debug.planner_errors, *(planner_errors or [])],
            "synthesis_errors": [*debug.synthesis_errors, *(synthesis_errors or [])],
            "llm_diagnostics": [*debug.llm_diagnostics, *(llm_diagnostics or [])],
            "latency_trace": [*debug.latency_trace, *latency_trace],
        }),
    }
