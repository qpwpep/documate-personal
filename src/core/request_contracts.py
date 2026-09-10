from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.answer_schema import AnswerDocument, AnswerResponse, iter_content_units
from src.core.evidence import EvidenceRef
from src.core.request_bodies import (
    AcknowledgeBody, AnswerReference, BodyPlan, BoundAnswerReference, BoundInputText, ComposeBody,
    CopyAnswerBody, CopyInputBody, ExtractBody, InputTextReference, TransformAnswerBody,
    TransformInputBody, UnresolvedBody, UserTurnSnapshot, WireBodyPlan,
    WireCopyAnswerBody, WireCopyInputBody, WireTransformAnswerBody, WireTransformInputBody,
    body_to_wire,
)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContractEvidence(ContractModel):
    """An interpreted user clause, not retrieved evidence or an execution permit."""

    id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    scope: str = Field(min_length=1, description="The affected field or subject, e.g. actions.save_text or answer.content.code_example.")
    interpretation: Literal["instruction", "negation", "mention", "quotation", "correction", "reference"]

    @field_validator("id", "turn_id", "quote", "scope")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("contract evidence must not be blank")
        return value


class ActionRequest(ContractModel):
    intent: Literal["requested", "forbidden", "not_requested", "unresolved"] = Field(default="not_requested", description="This turn's action instruction only. For a pending request, not_requested with no evidence means no change; the server preserves prior authorization and completion. A requested TXT file is save_text; merely reformatting text on screen is not saving.")
    evidence_ids: tuple[str, ...] = ()


class ActionContract(ContractModel):
    save_text: ActionRequest = Field(default_factory=ActionRequest)
    slack_notify: ActionRequest = Field(default_factory=ActionRequest)


class ContractDestination(ContractModel):
    channel_id: str | None = None
    user_id: str | None = None
    email: str | None = None

    @field_validator("channel_id", "user_id", "email")
    @classmethod
    def trim_optional(cls, value: str | None) -> str | None:
        return (value.strip() or None) if value is not None else None

    def has_destination(self) -> bool:
        return bool(self.channel_id or self.user_id or self.email)


RequirementMode = Literal["required", "forbidden", "preferred"]


class ContentRequirement(ContractModel):
    kind: Literal["code_example", "example", "comparison", "options_summary", "explanation", "procedure", "custom"] = Field(description="Semantic content only. Do not repeat the body transformation, action, line count or layout as custom content. A table prohibition belongs to format.table, not comparison or custom.")
    mode: RequirementMode
    description: str = ""
    evidence_ids: tuple[str, ...] = ()


class LayoutRequirement(ContractModel):
    kind: Literal["code_block", "ordered_list", "list", "table", "headings"] = Field(description="Explicit layout. No table means table/forbidden, not comparison or custom content. Layouts have no numeric value.")
    mode: RequirementMode
    value: None = None
    evidence_ids: tuple[str, ...] = ()


class LineCountRequirement(ContractModel):
    kind: Literal["line_count"] = "line_count"
    mode: RequirementMode
    value: int = Field(ge=1, strict=True, description="The explicit number of nonempty body lines. Never null.")
    evidence_ids: tuple[str, ...] = ()

    @field_validator("value", mode="before")
    @classmethod
    def json_integer_value(cls, value: Any) -> Any:
        # JSON Schema integer includes integral JSON numbers such as 1.0, but
        # never booleans or numeric strings. Preserve that boundary exactly.
        return int(value) if type(value) is float and value.is_integer() else value


FormatConstraint = LayoutRequirement | LineCountRequirement


def FormatRequirement(*, kind: str, mode: RequirementMode, value: int | None = None, evidence_ids: tuple[str, ...] = ()) -> FormatConstraint:
    """Keep the public constructor while typed branches define the actual schema."""
    model = LineCountRequirement if kind == "line_count" else LayoutRequirement
    return model(kind=kind, mode=mode, value=value, evidence_ids=evidence_ids)


class AnswerContract(ContractModel):
    """Content, presentation and soft preferences stay independent of retrieval."""

    content: tuple[ContentRequirement, ...] = ()
    format: tuple[FormatConstraint, ...] = ()
    preferences: tuple[str, ...] = ()

    @model_validator(mode="after")
    def reject_conflicts(self) -> "AnswerContract":
        for constraints in (self.content, self.format):
            modes: dict[tuple[str, Any], set[str]] = {}
            for item in constraints:
                key = (item.kind, getattr(item, "value", None))
                modes.setdefault(key, set()).add(item.mode)
            if any({"required", "forbidden"} <= states for states in modes.values()):
                raise ValueError("resolve conflicting required and forbidden constraints before finalizing")
        required_lines = {item.value for item in self.format if item.kind == "line_count" and item.mode == "required"}
        if len(required_lines) > 1:
            raise ValueError("resolve conflicting line counts before finalizing")
        aliases = {"code_example": "code", "code_block": "code"}
        required = {aliases.get(item.kind, item.kind) for item in (*self.content, *self.format) if item.mode == "required"}
        forbidden = {aliases.get(item.kind, item.kind) for item in (*self.content, *self.format) if item.mode == "forbidden"}
        implied = required | ({"example"} if "code" in required else set()) | ({"list"} if "ordered_list" in required else set())
        # A specific line count and a different forbidden count can coexist.
        if (implied & forbidden) - {"line_count"}:
            raise ValueError("content and format constraints conflict")
        return self


class MissingInformation(ContractModel):
    slot: Literal["subject", "input_reference", "answer_reference", "pending_request", "save_intent", "slack_intent", "slack_destination"] = Field(description="The actual missing user fact. save_intent/slack_intent mean whether to execute, not filename, path or destination. Save filenames are generated by the server and require no user input. A pure prohibition needs no subject or destination.")
    reason: Literal["not_provided", "unknown_id", "quote_not_found", "ambiguous_scope", "unclear"]
    question: str = Field(min_length=1)


class _RequestFacts(ContractModel):
    relation: Literal["new", "correction", "supplement", "cancel"] = "new"
    target_request_id: str | None = None
    actions: ActionContract = Field(default_factory=ActionContract)
    slack_destination: ContractDestination | None = None
    answer: AnswerContract = Field(default_factory=AnswerContract)
    evidence: tuple[ContractEvidence, ...] = ()
    missing_info: tuple[MissingInformation, ...] = ()

    @model_validator(mode="after")
    def validate_authority(self) -> "_RequestFacts":
        evidence = {item.id: item for item in self.evidence}
        if len(evidence) != len(self.evidence):
            raise ValueError("contract evidence ids must be unique")
        references = [self.actions.save_text, self.actions.slack_notify, *self.answer.content, *self.answer.format]
        if hasattr(self.body, "evidence_ids"):
            references.append(self.body)
        for item in references:
            if any(key not in evidence for key in item.evidence_ids):
                raise ValueError("unknown contract evidence id")
        def supports(ids: tuple[str, ...], path: str, allowed: set[str]) -> bool:
            return any(
                evidence[key].interpretation in allowed and (
                    evidence[key].scope == "current_request" or path == evidence[key].scope
                    or path.startswith(evidence[key].scope + ".")
                ) for key in ids
            )

        for name, action in (("save_text", self.actions.save_text), ("slack_notify", self.actions.slack_notify)):
            if action.intent in {"requested", "forbidden"}:
                if not action.evidence_ids:
                    raise ValueError("explicit actions require user evidence")
                allowed = {"instruction", "correction", "reference"} if action.intent == "requested" else {"negation", "correction", "instruction"}
                if not supports(action.evidence_ids, f"actions.{name}", allowed):
                    raise ValueError("actions require an instruction within their scope, not a mention or quotation")
        for group, requirements in (("content", self.answer.content), ("format", self.answer.format)):
            for requirement in requirements:
                allowed = {"negation", "correction", "instruction"} if requirement.mode == "forbidden" else {"instruction", "correction", "reference"}
                if not supports(requirement.evidence_ids, f"answer.{group}.{requirement.kind}", allowed):
                    raise ValueError("answer requirements require an instruction within their scope")
        if self.relation == "cancel" and any(
            action.intent == "requested" for action in (self.actions.save_text, self.actions.slack_notify)
        ):
            raise ValueError("cancellation cannot request an action")
        return self


class WireRequestContract(_RequestFacts):
    """Model interpretation: only user facts and offered reference selectors."""

    body: WireBodyPlan = Field(default_factory=ComposeBody)


class RequestContract(_RequestFacts):
    """Immutable server-bound facts; execution readiness is derived, never model asserted."""

    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    revision: int = Field(default=1, ge=1)
    body: BodyPlan = Field(default_factory=ComposeBody)
    body_request: WireBodyPlan | None = None
    failure: str | None = None

    @property
    def status(self) -> Literal["resolved", "unresolved", "invalid"]:
        if self.failure is not None:
            return "invalid"
        if not (self.can_prepare_body() or self.can_acknowledge() or self.can_cancel_pending()) or any(item.slot != "slack_destination" for item in self.missing_info):
            return "unresolved"
        return "resolved"

    @property
    def clarification_question(self) -> str | None:
        if isinstance(self.body, UnresolvedBody):
            return self.body.question
        return self.missing_info[0].question if self.missing_info else None

    def can_prepare_body(self) -> bool:
        return self.failure is None and self.relation != "cancel" and self.body.kind not in {"unresolved", "acknowledge"} and not any(
            item.slot in {"subject", "input_reference", "answer_reference", "pending_request"} for item in self.missing_info
        )

    def action_requested(self, kind: Literal["save_text", "slack_notify"]) -> bool:
        return getattr(self.actions, kind).intent == "requested"

    def execution_ready(self, kind: Literal["save_text", "slack_notify"], *, body_ready: bool, destination_ready: bool = False) -> bool:
        if not self.can_prepare_body() or not body_ready or not self.action_requested(kind):
            return False
        blocking = {"save_intent"} if kind == "save_text" else {"slack_intent"}
        if any(item.slot in blocking for item in self.missing_info):
            return False
        if kind == "slack_notify" and any(item.slot == "slack_destination" and item.reason != "not_provided" for item in self.missing_info):
            return False
        return kind != "slack_notify" or destination_ready

    def can_cancel_pending(self) -> bool:
        return (self.failure is None and self.relation == "cancel" and self.target_request_id is not None
                and not any(getattr(self.actions, kind).intent == "unresolved" for kind in ("save_text", "slack_notify"))
                and not any(item.slot in {"pending_request", "save_intent", "slack_intent"} for item in self.missing_info))

    def can_acknowledge(self) -> bool:
        return self.failure is None and self.body.kind == "acknowledge" and not self.missing_info and not any(
            getattr(self.actions, kind).intent in {"requested", "unresolved"} for kind in ("save_text", "slack_notify")
        )

    def to_wire(self) -> WireRequestContract:
        return WireRequestContract.model_validate({
            **self.model_dump(mode="python", exclude={"request_id", "revision", "body", "body_request", "failure"}),
            "body": self.body_request or body_to_wire(self.body),
        })

    @classmethod
    def invalid(cls, *, request_id: str | None = None, revision: int = 1) -> "RequestContract":
        return cls(request_id=request_id or str(uuid4()), revision=revision, failure="contract_invalid",
                   body=UnresolvedBody(question="요청의 실행 의사와 답변 조건을 해석하지 못했습니다. 다시 요청해 주세요."))


def validate_contract_evidence(contract: RequestContract | WireRequestContract, utterances: dict[str, str]) -> list[str]:
    """Verify provenance only. Quote presence does not prove correct interpretation."""
    errors = []
    for item in contract.evidence:
        if item.turn_id not in utterances:
            errors.append(f"unknown_turn_id:{item.id}")
        elif item.quote not in utterances[item.turn_id]:
            errors.append(f"quote_not_found:{item.id}")
    return errors


def resolve_body_response(
    contract: RequestContract, *, previous_response: AnswerResponse | None = None,
    pending_response: AnswerResponse | None = None,
) -> AnswerResponse | None:
    if not isinstance(contract.body, (CopyAnswerBody, TransformAnswerBody)):
        return None
    response = {"previous": previous_response, "pending": pending_response}.get(contract.body.source.ref)
    if response is None or response.content_hash != contract.body.source.response_hash:
        return None
    return response


def required_contract_turn_ids(contract: RequestContract) -> set[str]:
    ids = {item.turn_id for item in contract.evidence}
    if isinstance(contract.body, (CopyInputBody, TransformInputBody)):
        ids.add(contract.body.source.turn_id)
    if isinstance(contract.body_request, (WireCopyInputBody, WireTransformInputBody)):
        ids.add(contract.body_request.source.turn_id)
    return ids


@dataclass(slots=True)
class ContractCheck:
    missing_required: list[str] = field(default_factory=list)
    forbidden_present: list[str] = field(default_factory=list)
    unchecked_semantic: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.missing_required and not self.forbidden_present


def check_answer_contract(contract: AnswerContract, document: Any, *, evidence: Iterable[EvidenceRef] = ()) -> ContractCheck:
    """Check observable content; keep semantic evaluation explicitly separate.

    Line counts refer to nonblank body lines, excluding citation appendices and
    receipts. No user text is reinterpreted here.
    """
    document = document if isinstance(document, AnswerDocument) else AnswerDocument.model_validate(document)
    blocks = document.blocks
    units = [unit for _, unit in iter_content_units(document)]
    has_code = any(block.type == "code" for block in blocks) or any(
        re.search(r"(?m)^\s*(?:`{3,}|~{3,})", unit.text) for unit in units
    )
    code_sources = {item.id: item.excerpt.strip() for item in evidence if item.element.kind == "code"}
    has_code = has_code or any(
        unit.text.strip() == code_sources[reference]
        for unit in units for reference in unit.refs if reference in code_sources
    )
    # Paragraph units are displayed on one line unless their text has newlines.
    body_segments: list[str] = []
    for block in blocks:
        if block.type == "paragraph":
            body_segments.append(" ".join(unit.text for unit in block.content))
        elif block.type == "list":
            body_segments.extend(unit.text for unit in block.items)
        elif block.type == "table":
            body_segments.append(" | ".join(unit.text for unit in block.columns))
            body_segments.extend(" | ".join(unit.text for unit in row) for row in block.rows)
        else:
            body_segments.append(block.content.text)
    line_count = sum(bool(line.strip()) for segment in body_segments for line in segment.splitlines())
    present = {
        "code_example": has_code,
        "example": has_code or any(unit.basis == "example" for unit in units),
        "code_block": has_code,
        "ordered_list": any(block.type == "list" and block.ordered for block in blocks),
        "list": any(block.type == "list" for block in blocks),
        "table": any(block.type == "table" for block in blocks),
        "headings": any(block.type == "heading" for block in blocks),
    }
    check = ContractCheck()
    for requirement in (*contract.content, *contract.format):
        if requirement.mode == "preferred":
            continue
        key = requirement.kind
        if key == "line_count":
            found = line_count == requirement.value
            key = f"line_count:{requirement.value}"
        elif key in present:
            found = present[key]
        else:
            check.unchecked_semantic.append(f"{requirement.mode}:{key}")
            continue
        if requirement.mode == "required" and not found:
            check.missing_required.append(key)
        elif requirement.mode == "forbidden" and found:
            check.forbidden_present.append(key)
    check.missing_required = list(dict.fromkeys(check.missing_required))
    check.forbidden_present = list(dict.fromkeys(check.forbidden_present))
    return check


def missing_required_content(contract: AnswerContract, document: Any) -> list[str]:
    return check_answer_contract(contract, document).missing_required
