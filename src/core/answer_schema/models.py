from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from src.core.evidence import EvidenceRef


class AnswerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContentUnit(AnswerModel):
    """The exact displayed content and its proposed relationship to sources."""
    text: str = Field(min_length=1)
    basis: Literal["source", "inference", "example", "interaction", "excerpt"] = "interaction"
    refs: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be whitespace")
        return value

    @field_validator("refs")
    @classmethod
    def unique_refs(cls, value: list[str]) -> list[str]:
        if any(not ref.strip() for ref in value):
            raise ValueError("evidence references must not be blank")
        return list(dict.fromkeys(value))


class ParagraphBlock(AnswerModel):
    type: Literal["paragraph"] = "paragraph"
    content: list[ContentUnit] = Field(min_length=1)


class ListBlock(AnswerModel):
    type: Literal["list"] = "list"
    ordered: bool = False
    items: list[ContentUnit] = Field(min_length=1)


class CodeBlock(AnswerModel):
    type: Literal["code"] = "code"
    language: str = ""
    content: ContentUnit

    @field_validator("language")
    @classmethod
    def single_line_language(cls, value: str) -> str:
        if any(char in value for char in ("\n", "\r", "`")):
            raise ValueError("invalid code language")
        return value


class TableBlock(AnswerModel):
    type: Literal["table"] = "table"
    columns: list[ContentUnit] = Field(min_length=1)
    rows: list[list[ContentUnit]] = Field(default_factory=list)

    @model_validator(mode="after")
    def rectangular_table(self) -> "TableBlock":
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("every table row must match the column count")
        return self


class HeadingBlock(AnswerModel):
    type: Literal["heading"] = "heading"
    level: int = Field(default=2, ge=1, le=6)
    content: ContentUnit


AnswerBlock = Annotated[ParagraphBlock | ListBlock | CodeBlock | TableBlock | HeadingBlock, Field(discriminator="type")]


class AnswerDocument(AnswerModel):
    blocks: list[AnswerBlock] = Field(default_factory=list)


def iter_content_units(document: AnswerDocument) -> Iterator[tuple[str, ContentUnit]]:
    """The same reading order is used for checks, citations, UI and export."""
    for index, block in enumerate(document.blocks):
        prefix = f"b{index}"
        if isinstance(block, ParagraphBlock):
            for i, unit in enumerate(block.content):
                yield f"{prefix}.content.{i}", unit
        elif isinstance(block, ListBlock):
            for i, unit in enumerate(block.items):
                yield f"{prefix}.items.{i}", unit
        elif isinstance(block, (CodeBlock, HeadingBlock)):
            yield f"{prefix}.content", block.content
        elif isinstance(block, TableBlock):
            for i, unit in enumerate(block.columns):
                yield f"{prefix}.columns.{i}", unit
            for r, row in enumerate(block.rows):
                for c, unit in enumerate(row):
                    yield f"{prefix}.rows.{r}.{c}", unit


def document_hash(document: AnswerDocument) -> str:
    serialized = json.dumps(document.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class Citation(AnswerModel):
    number: int = Field(ge=1)
    evidence: EvidenceRef


class UnitCheck(AnswerModel):
    unit_id: str
    reference_status: Literal["resolved", "missing", "not_required"]
    support_status: Literal["not_evaluated", "exact_match", "unsupported"] = "not_evaluated"
    issues: list[str] = Field(default_factory=list)


def derive_checks_and_citations(
    document: AnswerDocument, evidence: list[EvidenceRef], *, retrieval_required: bool,
) -> tuple[list[UnitCheck], list[Citation]]:
    """Evaluate deterministic source checks against the exact displayed units.

    Resolving references does not evaluate a paraphrase's semantic support.
    Only an excerpt equal to one retained source selection gets exact_match.
    """
    by_id = {item.id: item for item in evidence}
    citations: list[Citation] = []
    adopted: set[str] = set()
    checks: list[UnitCheck] = []
    for path, unit in iter_content_units(document):
        missing = [ref for ref in unit.refs if ref not in by_id]
        requires_refs = unit.basis in {"source", "inference", "excerpt"} or (retrieval_required and unit.basis == "example")
        status = "missing" if missing or (requires_refs and not unit.refs) else ("resolved" if unit.refs else "not_required")
        messages = [f"unknown_evidence:{ref}" for ref in missing]
        if requires_refs and not unit.refs:
            messages.append("evidence_required")
        support = "not_evaluated"
        if unit.basis == "excerpt" and status == "resolved":
            if len(unit.refs) == 1 and unit.text == by_id[unit.refs[0]].excerpt:
                support = "exact_match"
            else:
                support = "unsupported"
                messages.append("excerpt_does_not_match_source")
        checks.append(UnitCheck(unit_id=path, reference_status=status, support_status=support, issues=messages))
        for ref in unit.refs:
            if ref in by_id and ref not in adopted:
                citations.append(Citation(number=len(citations) + 1, evidence=by_id[ref]))
                adopted.add(ref)
    return checks, citations


class ResponseIssue(AnswerModel):
    code: str
    message: str
    unit_id: str | None = None


class ActionReceipt(AnswerModel):
    kind: Literal["save_text", "slack_notify"]
    status: Literal["success", "error", "skipped"]
    file_path: str | None = None
    target: str | None = None
    message: str | None = None
    error: str | None = None


class AnswerResponse(AnswerModel):
    content: AnswerDocument = Field(default_factory=AnswerDocument)
    citations: list[Citation] = Field(default_factory=list)
    checks: list[UnitCheck] = Field(default_factory=list)
    issues: list[ResponseIssue] = Field(default_factory=list)
    actions: list[ActionReceipt] = Field(default_factory=list)
    content_hash: str = ""
    retrieval_required: bool = False

    @model_validator(mode="after")
    def consistent_revision(self) -> "AnswerResponse":
        expected = document_hash(self.content)
        if self.content.blocks and not self.content_hash:
            raise ValueError("nonempty responses must retain their checked content revision")
        if self.content_hash and self.content_hash != expected:
            raise ValueError("content changed after its checks were recorded")
        self.content_hash = expected
        ids = [citation.evidence.id for citation in self.citations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate citation evidence")
        if [c.number for c in self.citations] != list(range(1, len(self.citations) + 1)):
            raise ValueError("citation numbers must be consecutive in reading order")
        expected_checks, expected_citations = derive_checks_and_citations(
            self.content, [citation.evidence for citation in self.citations],
            retrieval_required=self.retrieval_required,
        )
        if self.checks != expected_checks:
            raise ValueError("checks must match every displayed content unit and its source references in reading order")
        if self.citations != expected_citations:
            raise ValueError("citations must contain only referenced sources in first-reference order")
        paths = {check.unit_id for check in self.checks}
        if any(issue.unit_id is not None and issue.unit_id not in paths for issue in self.issues):
            raise ValueError("response issue refers to an unknown content unit")
        return self
