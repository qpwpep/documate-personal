"""Preserve validation attempts even when a graph node fails before commit."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from src.core.contracts.validation_diagnostics import (
    DiagnosticCandidate, DiagnosticPacket, DiagnosticPredicate, ValidationDiagnostics, ValidationObservation,
)
from src.core.decisions import DecisionResult, prepared_packet_fingerprint
from src.core.evidence import EvidenceRef, SearchHit
from src.core.planner_schema import RetrievalTask


class ValidationDiagnosticsRecorder:
    def __init__(self) -> None:
        self.evidence: dict[str, EvidenceRef] = {}
        self.observations: list[ValidationObservation] = []

    def add_evidence(self, evidence: list[EvidenceRef]) -> None:
        for source in evidence:
            if source.id not in self.evidence:
                self.evidence[source.id] = source.model_copy(deep=True)

    def snapshot(self) -> ValidationDiagnostics:
        return ValidationDiagnostics(schema_version=1, evidence=self.evidence,
                                     observations=self.observations).model_copy(deep=True)


_recorder: ContextVar[ValidationDiagnosticsRecorder | None] = ContextVar("validation_diagnostics", default=None)
_current: ContextVar[ValidationObservation | None] = ContextVar("validation_observation", default=None)


@contextmanager
def capture_validation_diagnostics() -> Iterator[ValidationDiagnosticsRecorder]:
    recorder = ValidationDiagnosticsRecorder()
    token = _recorder.set(recorder)
    try:
        yield recorder
    finally:
        _recorder.reset(token)


def current_validation_diagnostics() -> ValidationDiagnostics:
    recorder = _recorder.get()
    return recorder.snapshot() if recorder is not None else ValidationDiagnostics(
        schema_version=1, evidence={}, observations=[],
    )


@contextmanager
def record_validation(observation: ValidationObservation) -> Iterator[ValidationObservation]:
    recorder = _recorder.get()
    if recorder is None:
        yield observation
        return
    observation.sequence = len(recorder.observations) + 1
    recorder.observations.append(observation)
    token = _current.set(observation)
    try:
        yield observation
    except BaseException as exc:
        observation.status = "failed"
        observation.failure_type = type(exc).__name__
        raise
    finally:
        if observation.status == "started":
            observation.status = "completed"
        _current.reset(token)


def record_candidates(hits: list[SearchHit]) -> None:
    recorder, observation = _recorder.get(), _current.get()
    if recorder is None or observation is None:
        return
    recorder.add_evidence([hit.evidence for hit in hits])
    recorded = {(item.requirement_id, item.evidence_id, item.role): item for item in observation.candidates}
    for hit in hits:
        key = (hit.requirement_id, hit.evidence.id, hit.role)
        if key in recorded:
            recorded[key].supporting_evidence_ids = list(hit.supporting_evidence_ids)
            continue
        candidate = DiagnosticCandidate(
            requirement_id=hit.requirement_id, evidence_id=hit.evidence.id,
            role=hit.role, supporting_evidence_ids=list(hit.supporting_evidence_ids),
        )
        recorded[key] = candidate
        observation.candidates.append(candidate)


def record_packet(mode: Literal["normal", "compact", "used"], evidence: list[EvidenceRef], requirement_map: dict[str, list[str]],
                  failed_requirement_ids: list[str] | None = None) -> None:
    recorder, observation = _recorder.get(), _current.get()
    if recorder is None or observation is None:
        return
    recorder.add_evidence(evidence)
    packet = DiagnosticPacket(mode=mode, evidence_ids=[item.id for item in evidence],
        requirement_map={key: list(value) for key, value in requirement_map.items()},
        fingerprint=prepared_packet_fingerprint(evidence, requirement_map),
        failed_requirement_ids=list(failed_requirement_ids or []))
    observation.packets = [item for item in observation.packets if item.mode != mode] + [packet]


def register_predicates(predicates: list[DiagnosticPredicate]) -> None:
    observation = _current.get()
    if observation is not None:
        names = {item.name for item in observation.predicates}
        observation.predicates.extend(item for item in predicates if item.name not in names)


def record_decisions(result: DecisionResult, *, cached_names: set[str] | None = None) -> None:
    observation = _current.get()
    if observation is None:
        return
    cached = cached_names or set()
    for predicate in observation.predicates:
        if predicate.name in result.probabilities:
            predicate.status = "evaluated"
            predicate.probability = result.probabilities[predicate.name]
            predicate.cached = predicate.name in cached
        elif predicate.name in result.refused_names:
            predicate.status = "refused"
            predicate.cached = predicate.name in cached


def mark_validation_skipped(reason: str) -> None:
    observation = _current.get()
    if observation is not None:
        observation.status = "skipped"
        observation.skip_reason = reason


def current_validation_observation() -> ValidationObservation | None:
    return _current.get()


def record_semantic_tasks(tasks: list[RetrievalTask]) -> None:
    observation = _current.get()
    if observation is not None:
        observation.semantic_tasks = [task.model_copy(deep=True) for task in tasks]


def record_candidate_context(requirement_id: str, evidence_id: str, context_ids: list[str]) -> None:
    observation = _current.get()
    if observation is not None:
        for candidate in observation.candidates:
            if candidate.requirement_id == requirement_id and candidate.evidence_id == evidence_id:
                candidate.context_evidence_ids = list(context_ids)


def validation_diagnostics_enabled() -> bool:
    return _recorder.get() is not None
