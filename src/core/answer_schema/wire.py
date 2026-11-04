"""Public answers retain each captured source once, independently of citations."""
from __future__ import annotations

import hashlib
import json
from typing import Annotated

from pydantic import BeforeValidator, Field, PlainSerializer

from src.core.answer_schema.models import (
    ActionReceipt, AnswerDocument, AnswerModel, AnswerResponse, Citation,
    ResponseIssue, SemanticValidation, UnitCheck,
)
from src.core.documents import DocumentElement, DocumentSnapshot
from src.core.evidence import EvidenceRef, SourceSelection


class CapturedAnswerSource(AnswerModel):
    snapshot: DocumentSnapshot
    element: DocumentElement

    def source_id(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return "source:" + hashlib.sha256(encoded).hexdigest()


class SourceCitation(AnswerModel):
    number: int = Field(ge=1)
    evidence_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    selection: SourceSelection


class SourcedAnswerPayload(AnswerModel):
    content: AnswerDocument
    sources: dict[str, CapturedAnswerSource]
    citations: list[SourceCitation]
    checks: list[UnitCheck]
    issues: list[ResponseIssue]
    actions: list[ActionReceipt]
    content_hash: str
    retrieval_required: bool
    semantic_validation: SemanticValidation | None

    @classmethod
    def from_answer(cls, answer: AnswerResponse) -> SourcedAnswerPayload:
        sources: dict[str, CapturedAnswerSource] = {}
        citations = []
        for citation in answer.citations:
            evidence = citation.evidence
            source = CapturedAnswerSource(snapshot=evidence.snapshot, element=evidence.element)
            source_id = source.source_id()
            sources.setdefault(source_id, source)
            citations.append(SourceCitation(
                number=citation.number, evidence_id=evidence.id,
                source_id=source_id, selection=evidence.selection,
            ))
        return cls(
            content=answer.content, sources=sources, citations=citations,
            checks=answer.checks, issues=answer.issues, actions=answer.actions,
            content_hash=answer.content_hash, retrieval_required=answer.retrieval_required,
            semantic_validation=answer.semantic_validation,
        )

    def to_answer(self) -> AnswerResponse:
        referenced = {citation.source_id for citation in self.citations}
        if referenced != self.sources.keys():
            raise ValueError("answer sources must match the cited source IDs")
        if any(source_id != source.source_id() for source_id, source in self.sources.items()):
            raise ValueError("answer source ID does not match its captured content")
        citations = [Citation(
            number=citation.number,
            evidence=EvidenceRef(
                id=citation.evidence_id,
                snapshot=self.sources[citation.source_id].snapshot,
                element=self.sources[citation.source_id].element,
                selection=citation.selection,
            ),
        ) for citation in self.citations]
        return AnswerResponse(
            content=self.content, citations=citations, checks=self.checks,
            issues=self.issues, actions=self.actions, content_hash=self.content_hash,
            retrieval_required=self.retrieval_required, semantic_validation=self.semantic_validation,
        )


def _parse_serialized_answer(value: object) -> AnswerResponse:
    if isinstance(value, AnswerResponse):
        return value
    return SourcedAnswerPayload.model_validate(value).to_answer()


SerializedAnswer = Annotated[
    AnswerResponse,
    BeforeValidator(_parse_serialized_answer, json_schema_input_type=SourcedAnswerPayload),
    PlainSerializer(SourcedAnswerPayload.from_answer, return_type=SourcedAnswerPayload, when_used="json"),
]
