from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from src.core.answer_schema import ActionReceipt, AnswerResponse

from ..decisions import CaseDecision, raw_results_fingerprint, validate_run_decision_evidence
from ..io import dump_jsonl
from ..result_models import CaseResult, JudgeSubscores, SaveAssessment
from ..summary_models import RunSummary
from ..tool_policy import ToolPolicyAssessment, ToolPolicySpec
from .markdown import build_markdown_report


@dataclass(frozen=True)
class ReportInputs:
    """Keep historical observations separate from current decision projections."""

    summary: RunSummary
    current_results: list[CaseResult] | None = None
    historical_results: list[dict[str, Any]] | None = None
    historical_summary: dict[str, Any] | None = None


class _HistoricalCaseIdentity(BaseModel):
    run_id: str
    case_id: str
    category: str
    query: str
    session_id: str
    endpoint: str
    request_payload: dict[str, Any]
    http_status: int
    created_at_utc: str


class _DecisionTurnEvidence(BaseModel):
    """Only the retained facts used by decision contract 1; no usage decoding."""

    request_id: str | None
    response: AnswerResponse | None
    execution_evidence: dict[str, Any] | None
    debug: dict[str, Any] | None
    tool_calls: list[str]


class _SavedCaseDecisionEvidence(_HistoricalCaseIdentity):
    decision_contract_version: Literal[1]
    policy_snapshot: ToolPolicySpec | None
    policy_assessment: ToolPolicyAssessment
    decision: CaseDecision
    release_pass: bool
    request_id: str | None
    execution_evidence: dict[str, Any] | None
    scenario_turns: list[_DecisionTurnEvidence]
    actions: list[ActionReceipt]
    retrieval_diagnostics: list[dict[str, Any]]
    debug: dict[str, Any] | None
    tool_calls: list[str]
    tool_call_count: int
    gate_failures: list[str]
    runtime_errors: list[str]
    response_errors: list[str]
    eval_validity: Literal["valid", "incomplete", "invalid"]
    invalid_eval: bool
    product_pass: bool | None
    judge_pass: bool | None
    judge_status: Literal["disabled", "not_run", "failed", "succeeded"]
    llm_judge_score: float | None
    judge_subscores: JudgeSubscores | None
    save_assessment: SaveAssessment | None
    setup_save_assessments: dict[int, SaveAssessment]

    @model_validator(mode="after")
    def validate_judge_evidence(self) -> "_SavedCaseDecisionEvidence":
        judge_values = (self.llm_judge_score, self.judge_subscores, self.judge_pass)
        if self.judge_status == "succeeded":
            if any(value is None for value in judge_values):
                raise ValueError("a saved successful judge requires its score and verdict")
        elif any(value is not None for value in judge_values):
            raise ValueError("a saved unsuccessful judge cannot carry a score or verdict")
        if self.invalid_eval != (self.eval_validity == "invalid"):
            raise ValueError("saved invalid_eval contradicts its evaluation validity")
        return self


def is_current_measurement(summary: RunSummary) -> bool:
    from .summary import MEASUREMENT_CONTRACT_VERSION

    return summary.measurement_contract_version == MEASUREMENT_CONTRACT_VERSION


def write_run_outputs(*, output_dir: Path, results: list[CaseResult], summary: RunSummary) -> None:
    from ..decisions import validate_run_outputs

    validate_run_outputs(summary, results)
    report = build_markdown_report(summary, results)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_dir / "raw_results.jsonl", results)
    summary_payload = summary.model_dump(exclude={"weights"} if summary.weights is None else set())
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    dump_jsonl(
        output_dir / "request_map.jsonl",
        [
            {
                "run_id": result.run_id,
                "case_id": result.case_id,
                "session_id": result.session_id,
                "request_id": result.request_id,
                "query": result.query[:240],
                "query_length": len(result.query),
                "query_hash": hashlib.sha256(result.query.encode("utf-8")).hexdigest(),
                "trace": result.trace,
                "created_at_utc": result.created_at_utc,
            }
            for result in results
        ],
    )


def load_run_outputs(output_dir: Path) -> tuple[RunSummary, list[CaseResult]]:
    """Read a current run and verify its saved evidence without calling providers."""
    from ..decisions import validate_run_outputs

    _, raw_results, summary = _read_saved_payloads(output_dir)
    if not is_current_measurement(summary):
        raise ValueError("historical results cannot establish current release eligibility (measurement contract)")
    results = [CaseResult.model_validate(payload) for payload in raw_results]
    validate_run_outputs(summary, results)
    return summary, results


def _read_saved_payloads(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], RunSummary]:
    """Verify the original fingerprint before any defaults or typed projection."""
    summary_path = output_dir / "summary.json"
    raw_path = output_dir / "raw_results.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_results.jsonl not found: {raw_path}")
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_payload, dict):
        raise ValueError("run summary must be a JSON object")
    raw_results: list[dict[str, Any]] = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("case results must be JSON objects")
        raw_results.append(payload)
    if summary_payload.get("decision_contract_version") is not None:
        if summary_payload.get("results_fingerprint") != raw_results_fingerprint(raw_results):
            raise ValueError("run results fingerprint mismatch")
    summary = RunSummary.model_validate(summary_payload)
    return summary_payload, raw_results, summary


def load_report_inputs(output_dir: Path) -> ReportInputs:
    """Read saved facts without upgrading historical measurements or eligibility."""
    from ..decisions import validate_run_outputs

    summary_payload, raw_results, summary = _read_saved_payloads(output_dir)
    if is_current_measurement(summary):
        results = [CaseResult.model_validate(payload) for payload in raw_results]
        validate_run_outputs(summary, results)
        return ReportInputs(summary=summary, current_results=results)
    if summary.measurement_contract_version not in {None, "attachment-question-scenario-v1"}:
        raise ValueError("unsupported historical measurement contract")
    if summary.decision_contract_version is not None:
        evidence = [_SavedCaseDecisionEvidence.model_validate(payload) for payload in raw_results]
        validate_run_decision_evidence(summary, evidence)
        return ReportInputs(summary=summary, historical_results=raw_results, historical_summary=summary_payload)

    case_ids: set[str] = set()
    for payload in raw_results:
        if payload.get("decision_contract_version") is not None or any(payload.get(field) is not None for field in (
            "policy_snapshot", "policy_assessment", "decision", "execution_evidence",
        )):
            raise ValueError("historical run contains versioned decision or policy evidence")
        parsed = _HistoricalCaseIdentity.model_validate(payload)
        if parsed.run_id != summary.run_id:
            raise ValueError("historical case result belongs to a different run")
        case_ids.add(parsed.case_id)
    if len(raw_results) != summary.metrics.total_cases:
        raise ValueError("historical result count does not match the saved summary")
    duplicates = len(raw_results) - len(case_ids)
    if duplicates != summary.metrics.duplicate_result_cases:
        raise ValueError("historical duplicate count does not match the saved summary")
    return ReportInputs(summary=summary, historical_results=raw_results, historical_summary=summary_payload)


__all__ = [
    "ReportInputs",
    "load_report_inputs",
    "load_run_outputs",
    "is_current_measurement",
    "write_run_outputs",
]
