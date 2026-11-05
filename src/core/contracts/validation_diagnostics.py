"""Recorded validation inputs and judgments, independent of product decisions."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.answer_schema import AnswerDocument
from src.core.answer_schema.models import document_hash
from src.core.evidence import EvidenceRef
from src.core.decisions import prepared_packet_fingerprint
from src.core.planner_schema import RetrievalTask
from src.core.request_contracts import RequestContract


class DiagnosticModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiagnosticCandidate(DiagnosticModel):
    requirement_id: str
    evidence_id: str
    role: Literal["candidate", "context"]
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    context_evidence_ids: list[str] = Field(default_factory=list)


class DiagnosticPacket(DiagnosticModel):
    mode: Literal["normal", "compact", "used"]
    evidence_ids: list[str]
    requirement_map: dict[str, list[str]]
    fingerprint: str
    failed_requirement_ids: list[str] = Field(default_factory=list)


class DiagnosticPredicate(DiagnosticModel):
    name: str
    criterion: str
    requirement_id: str | None = None
    evidence_id: str | None = None
    unit_id: str | None = None
    packet_mode: Literal["normal", "compact", "used"] | None = None
    status: Literal["pending", "evaluated", "refused"] = "pending"
    probability: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    cached: bool = False

    @model_validator(mode="after")
    def probability_matches_status(self) -> DiagnosticPredicate:
        if (self.status == "evaluated") != (self.probability is not None):
            raise ValueError("only evaluated predicates have a probability")
        return self


class ValidationObservation(DiagnosticModel):
    sequence: int = Field(ge=1)
    stage: Literal["pre_synthesis_validation", "post_synthesis_validation"]
    attempt: int = Field(ge=1)
    revision: Literal["preparation", "generated", "filtered"]
    model: str
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    status: Literal["started", "completed", "skipped", "failed"] = "started"
    user_input: str
    request_contract: RequestContract | None
    tasks: list[RetrievalTask]
    semantic_tasks: list[RetrievalTask] = Field(default_factory=list)
    candidates: list[DiagnosticCandidate] = Field(default_factory=list)
    packets: list[DiagnosticPacket] = Field(default_factory=list)
    selected_packet_mode: Literal["normal", "compact"] | None = None
    body: AnswerDocument | None = None
    body_hash: str | None = None
    terminal_failed_requirement_ids: list[str] = Field(default_factory=list)
    predicates: list[DiagnosticPredicate] = Field(default_factory=list)
    local_failures: list[str] = Field(default_factory=list)
    local_failed_unit_ids: list[str] = Field(default_factory=list)
    failed_requirement_ids: list[str] = Field(default_factory=list)
    skip_reason: str | None = None
    failure_type: str | None = None

    @model_validator(mode="after")
    def body_identity_matches(self) -> ValidationObservation:
        if self.body is None:
            if self.body_hash is not None:
                raise ValueError("a body hash requires the observed body")
        elif self.body_hash != document_hash(self.body):
            raise ValueError("observed body hash does not match its content")
        if len({item.name for item in self.predicates}) != len(self.predicates):
            raise ValueError("duplicate predicate name in validation observation")
        return self


class ValidationDiagnostics(DiagnosticModel):
    schema_version: Literal[1]
    evidence: dict[str, EvidenceRef]
    observations: list[ValidationObservation]

    @model_validator(mode="after")
    def references_are_recorded(self) -> ValidationDiagnostics:
        if any(identity != source.id for identity, source in self.evidence.items()):
            raise ValueError("validation evidence registry identity mismatch")
        for sequence, observation in enumerate(self.observations, 1):
            if observation.sequence != sequence:
                raise ValueError("validation observation sequence is not contiguous")
            referenced = {identity for packet in observation.packets for identity in packet.evidence_ids}
            referenced.update(item.evidence_id for item in observation.candidates)
            referenced.update(identity for item in observation.candidates for identity in item.supporting_evidence_ids)
            referenced.update(identity for item in observation.candidates for identity in item.context_evidence_ids)
            referenced.update(item.evidence_id for item in observation.predicates if item.evidence_id is not None)
            if referenced - self.evidence.keys():
                raise ValueError("validation observation references unrecorded evidence")
            for packet in observation.packets:
                if packet.requirement_map.keys() - set(packet.evidence_ids):
                    raise ValueError("validation packet maps evidence outside its selection")
                if packet.fingerprint != prepared_packet_fingerprint(
                    [self.evidence[identity] for identity in packet.evidence_ids], packet.requirement_map,
                ):
                    raise ValueError("validation packet fingerprint mismatch")
        return self
