"""Read-only validation diagnosis and deterministic independent-review samples."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.contracts.validation_diagnostics import ValidationDiagnostics, ValidationObservation
from src.core.evidence import EvidenceRef
from .config_models import BenchmarkCase
from .decisions import raw_results_fingerprint
from .diagnostic_review import DiagnosticReviewOutcome


DiagnosticCause = Literal[
    "diagnostics_unobserved", "candidate_gap", "selection_gap", "semantic_rejection",
    "predicate_refusal", "body_condition_failure", "local_validation_failure",
    "unassessed_requirements", "technical_failure",
]
OracleStatus = Literal["verified", "not_available", "missing_file", "hash_mismatch"]


class DiagnosticCaseSummary(BaseModel):
    case_id: str
    category: str
    original_product_pass: bool | None
    original_judge_pass: bool | None
    original_release_pass: bool | None
    causes: list[DiagnosticCause]
    observed_turns: int
    unobserved_turns: int


class DiagnosticReviewBundle(BaseModel):
    """Unapproved candidate and its original inputs, never a replacement answer."""

    scope: Literal["unapproved_validation_sample"] = "unapproved_validation_sample"
    sample_id: str
    case_id: str
    category: str
    turn_role: Literal["setup", "question"]
    turn_index: int
    execution_id: str | None
    query: str
    causes: list[DiagnosticCause]
    original_product_pass: bool | None
    original_judge_pass: bool | None
    original_release_pass: bool | None
    oracle_status: OracleStatus
    case: BenchmarkCase | None
    observation: ValidationObservation
    evidence: dict[str, EvidenceRef]


class DiagnosticJudgment(BaseModel):
    sample_id: str
    scope: Literal["diagnostic_only"] = "diagnostic_only"
    outcome: DiagnosticReviewOutcome


class DiagnosticReport(BaseModel):
    schema_version: Literal[1] = 1
    scope: Literal["diagnostic_only"] = "diagnostic_only"
    source_run_id: str
    source_results_fingerprint: str
    sample_per_cause: int
    sampling_policy: Literal["sha256-case-turn-observation-v1"] = "sha256-case-turn-observation-v1"
    cause_count_unit: Literal["cases_with_observed_signal"] = "cases_with_observed_signal"
    cause_counts: dict[str, int]
    oracle_status: OracleStatus
    cases: list[DiagnosticCaseSummary]
    reviews: list[DiagnosticReviewBundle]
    diagnostic_judgments: list[DiagnosticJudgment]


class _SavedDebug(BaseModel):
    validation_diagnostics: ValidationDiagnostics | None = None
    missing_required_debug_fields: list[str] = Field(default_factory=list)


class _SavedTurn(BaseModel):
    role: Literal["setup", "question"] = "question"
    query: str
    execution_id: str | None = None
    debug: _SavedDebug | None = None


class _SavedCase(BaseModel):
    """Project only diagnostic fields; historical release contracts stay untouched."""

    model_config = ConfigDict(strict=True)
    run_id: str
    case_id: str
    category: str
    query: str
    execution_id: str | None = None
    debug: _SavedDebug | None = None
    scenario_turns: list[_SavedTurn] = Field(default_factory=list)
    product_pass: bool | None = None
    judge_pass: bool | None = None
    release_pass: bool | None = None


class _SavedSummary(BaseModel):
    run_id: str
    results_fingerprint: str


class _DatasetApproval(BaseModel):
    status: str
    candidate_sha256: str | None = None


class _SavedManifest(BaseModel):
    status: Literal["running", "completed", "interrupted", "failed"]
    fixtures_path: str | None = None
    dataset_approval: _DatasetApproval | None = None


def diagnose_run(
    run_dir: Path, output_path: Path, *, sample_per_cause: int = 2,
    judge: Callable[[DiagnosticReviewBundle], DiagnosticReviewOutcome] | None = None,
) -> DiagnosticReport:
    """Export observed failure signals without regrading or modifying the source run.

    A supplied judge reviews only sampled bodies. Its outcomes are kept in a
    separate diagnostic field and never change case or release decisions.
    """
    if type(sample_per_cause) is not int or sample_per_cause < 1:
        raise ValueError("sample_per_cause must be a positive integer")
    source_dir = run_dir.resolve()
    destination = output_path.resolve()
    if destination.is_relative_to(source_dir):
        raise ValueError("diagnostic output must be outside the source run directory")
    if destination.exists():
        raise FileExistsError(destination)
    summary, results, manifest = _load_saved_run(source_dir)
    cases_by_id, oracle_status = _load_verified_cases(source_dir, manifest)
    case_summaries = []
    candidates = []
    for result in results:
        causes: set[DiagnosticCause] = set()
        observed_turns = 0
        unobserved_turns = 0
        turns = result.scenario_turns or [_SavedTurn(
            query=result.query, execution_id=result.execution_id, debug=result.debug,
        )]
        for index, turn in enumerate(turns):
            diagnostics = turn.debug.validation_diagnostics if turn.debug is not None else None
            if diagnostics is None or "validation_diagnostics" in turn.debug.missing_required_debug_fields:
                causes.add("diagnostics_unobserved")
                unobserved_turns += 1
                continue
            observed_turns += 1
            for observation in diagnostics.observations:
                signals = classify_observation(observation)
                causes.update(signals)
                if not signals:
                    continue
                fixture = cases_by_id.get(result.case_id)
                if fixture is not None and fixture.query != result.query:
                    raise ValueError(f"verified fixture query differs from recorded case: {result.case_id}")
                candidates.append(DiagnosticReviewBundle(
                    sample_id=f"{result.case_id}:{turn.role}:{index}:{observation.sequence}",
                    case_id=result.case_id, category=result.category, turn_role=turn.role,
                    turn_index=index, execution_id=turn.execution_id, query=turn.query,
                    causes=signals, original_product_pass=result.product_pass,
                    original_judge_pass=result.judge_pass, original_release_pass=result.release_pass,
                    oracle_status=("not_available" if oracle_status == "verified" and fixture is None else oracle_status),
                    case=fixture, observation=observation,
                    evidence=_observation_evidence(observation, diagnostics),
                ))
        case_summaries.append(DiagnosticCaseSummary(
            case_id=result.case_id, category=result.category, original_product_pass=result.product_pass,
            original_judge_pass=result.judge_pass, original_release_pass=result.release_pass,
            causes=sorted(causes), observed_turns=observed_turns, unobserved_turns=unobserved_turns,
        ))
    reviews = select_review_samples(candidates, sample_per_cause=sample_per_cause)
    judgments = []
    if judge is not None:
        for review in reviews:
            if review.observation.body is None:
                continue
            outcome = judge(review.model_copy(deep=True))
            judgments.append(DiagnosticJudgment(
                sample_id=review.sample_id, outcome=outcome,
            ))
    report = DiagnosticReport(
        source_run_id=summary.run_id, source_results_fingerprint=summary.results_fingerprint,
        sample_per_cause=sample_per_cause, oracle_status=oracle_status,
        cause_counts=dict(sorted(Counter(cause for case in case_summaries for cause in case.causes).items())),
        cases=case_summaries, reviews=reviews, diagnostic_judgments=judgments,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report.model_dump_json(indent=2) + "\n")
    return report


def classify_observation(observation: ValidationObservation) -> list[DiagnosticCause]:
    """Classify recorded signals, not hypothesized semantic mistakes or root causes."""
    causes: set[DiagnosticCause] = set()
    if observation.status == "failed" or observation.failure_type is not None:
        causes.add("technical_failure")
    if any(predicate.status == "pending" for predicate in observation.predicates) or (
        observation.status in {"started", "skipped"}
        and (observation.failed_requirement_ids or observation.terminal_failed_requirement_ids)
    ):
        causes.add("unassessed_requirements")
    if observation.local_failures:
        causes.add("local_validation_failure")
    if observation.stage == "pre_synthesis_validation" and observation.status == "completed":
        # Context explains a primary candidate's bindings; it does not itself
        # establish that the requested passage reached candidate preparation.
        candidates = {candidate.requirement_id for candidate in observation.candidates if candidate.role == "candidate"}
        selected_packets = [packet for packet in observation.packets
                            if packet.mode == (observation.selected_packet_mode or "normal")]
        for task in observation.tasks:
            if task.requirement_id not in candidates:
                causes.add("candidate_gap")
            elif selected_packets and all(
                task.requirement_id not in [identity for ids in packet.requirement_map.values() for identity in ids]
                for packet in selected_packets
            ):
                causes.add("selection_gap")
    assessed_requirements = set()
    for predicate in observation.predicates:
        if predicate.status == "refused":
            causes.add("predicate_refusal")
        if predicate.status != "evaluated":
            continue
        if predicate.requirement_id is not None:
            assessed_requirements.add(predicate.requirement_id)
        if predicate.probability is not None and predicate.probability < observation.threshold:
            if predicate.criterion in {"request_fulfilled", "format_fulfilled"} or predicate.criterion.startswith("content_condition"):
                causes.add("body_condition_failure")
            else:
                causes.add("semantic_rejection")
    if observation.stage == "post_synthesis_validation" and (
        set(observation.terminal_failed_requirement_ids) - assessed_requirements
    ):
        causes.add("unassessed_requirements")
    return sorted(causes)


def select_review_samples(
    candidates: list[DiagnosticReviewBundle], *, sample_per_cause: int,
) -> list[DiagnosticReviewBundle]:
    """Union N examples chosen independently per signal, without input-order effects."""
    ordered = sorted(candidates, key=lambda item: (
        hashlib.sha256(item.sample_id.encode("utf-8")).hexdigest(), item.sample_id,
    ))
    selected = {}
    for cause in sorted({cause for item in candidates for cause in item.causes}):
        matching = [item for item in ordered if cause in item.causes]
        for item in matching[:sample_per_cause]:
            selected[item.sample_id] = item
    return sorted(selected.values(), key=lambda item: item.sample_id)


def _load_saved_run(run_dir: Path) -> tuple[_SavedSummary, list[_SavedCase], _SavedManifest | None]:
    summary = _SavedSummary.model_validate_json((run_dir / "summary.json").read_text(encoding="utf-8"))
    raw = []
    for line in (run_dir / "raw_results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("case result must be a JSON object")
        raw.append(value)
    if summary.results_fingerprint != raw_results_fingerprint(raw):
        raise ValueError("run results fingerprint mismatch")
    results = [_SavedCase.model_validate(value) for value in raw]
    if any(result.run_id != summary.run_id for result in results):
        raise ValueError("case result run ID differs from summary")
    if len({result.case_id for result in results}) != len(results):
        raise ValueError("duplicate case results cannot be diagnosed as separate cases")
    manifest_path = run_dir / "manifest.json"
    manifest = (_SavedManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else None)
    if manifest is not None and manifest.status != "completed":
        raise ValueError("diagnosis requires a completed benchmark run")
    return summary, results, manifest


def _load_verified_cases(
    run_dir: Path, manifest: _SavedManifest | None,
) -> tuple[dict[str, BenchmarkCase], OracleStatus]:
    if (manifest is None or manifest.fixtures_path is None or manifest.dataset_approval is None
            or manifest.dataset_approval.status != "verified" or manifest.dataset_approval.candidate_sha256 is None):
        return {}, "not_available"
    path = Path(manifest.fixtures_path)
    if not path.is_absolute():
        # Recorded fixture paths are repository-relative. Resolve from the
        # existing project root, never from an arbitrary caller working folder.
        roots = [parent for parent in (run_dir, *run_dir.parents) if (parent / "pyproject.toml").is_file()]
        path = (roots[0] / path) if roots else (run_dir / path)
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return {}, "missing_file"
    if hashlib.sha256(content).hexdigest() != manifest.dataset_approval.candidate_sha256:
        return {}, "hash_mismatch"
    cases = [BenchmarkCase.model_validate_json(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("verified fixtures contain duplicate case IDs")
    return {case.case_id: case for case in cases}, "verified"


def _observation_evidence(
    observation: ValidationObservation, diagnostics: ValidationDiagnostics,
) -> dict[str, EvidenceRef]:
    identities = {candidate.evidence_id for candidate in observation.candidates}
    identities.update(identity for candidate in observation.candidates for identity in candidate.supporting_evidence_ids)
    identities.update(identity for candidate in observation.candidates for identity in candidate.context_evidence_ids)
    identities.update(identity for packet in observation.packets for identity in packet.evidence_ids)
    identities.update(predicate.evidence_id for predicate in observation.predicates if predicate.evidence_id is not None)
    return {identity: diagnostics.evidence[identity] for identity in sorted(identities)}
