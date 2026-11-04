"""Body requests distinguish model-selected references from server-bound data."""
from __future__ import annotations

import hashlib
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


BODY_INSTRUCTION_DESCRIPTION = (
    "Describe only the requested operation on the displayed body. Put file saving and message delivery "
    "in actions, not in this instruction. Preserve words about "
    "saving or sending when they are quoted payload or the subject being explained."
)


class BodyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UserTurnSnapshot(BodyModel):
    turn_id: str = Field(min_length=1)
    text: str


class InputTextReference(BodyModel):
    turn_id: str = Field(min_length=1)
    quote: str = Field(min_length=1, description="Exact payload text to copy or transform. Exclude surrounding quotation delimiters unless those marks themselves are part of the requested payload. Never paraphrase or normalize the source.")
    occurrence: int | None = Field(default=None, ge=1, description="One-based occurrence, only when the user's selection identifies it. Null means the quote must be unique.")


class BoundInputText(BodyModel):
    turn_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    content_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_selection(self) -> "BoundInputText":
        if self.end - self.start != len(self.text):
            raise ValueError("input offsets use Python Unicode code points and must match the selected text")
        if self.content_hash != hashlib.sha256(self.text.encode("utf-8")).hexdigest():
            raise ValueError("selected input hash does not match its text")
        return self


class AnswerReference(BodyModel):
    ref: Literal["previous", "pending"]


class OrdinalAnswerReference(BodyModel):
    ref: Literal["ordinal"]
    ordinal: int = Field(ge=1, description="The one-based answer position explicitly requested by the user. Preserve it even when no offered answer has this position; never substitute previous.")


class BoundAnswerReference(BodyModel):
    ref: Literal["previous", "pending"]
    response_hash: str = Field(min_length=1)


class ComposeBody(BodyModel):
    kind: Literal["compose"] = "compose"
    instruction: str = Field(default="", description=BODY_INSTRUCTION_DESCRIPTION)
    evidence_ids: tuple[str, ...] = ()


class AcknowledgeBody(BodyModel):
    """Confirm a complete instruction without creating a deliverable document."""
    kind: Literal["acknowledge"] = "acknowledge"
    evidence_ids: tuple[str, ...] = ()


class WireCopyInputBody(BodyModel):
    """Copy literal input with unchanged layout; a newly requested layout uses transform_input."""
    kind: Literal["copy_input"] = "copy_input"
    source: InputTextReference


class CopyInputBody(BodyModel):
    kind: Literal["copy_input"] = "copy_input"
    source: BoundInputText


class WireTransformInputBody(BodyModel):
    kind: Literal["transform_input"] = "transform_input"
    source: InputTextReference
    instruction: str = Field(min_length=1, description=(
        "The requested translation, edit or layout-only change. If wording must stay literal while adding "
        "a code block/list/table, preserve the selected payload and change only its presentation. "
        + BODY_INSTRUCTION_DESCRIPTION
    ))
    evidence_ids: tuple[str, ...] = ()


class TransformInputBody(BodyModel):
    kind: Literal["transform_input"] = "transform_input"
    source: BoundInputText
    instruction: str = Field(min_length=1, description=BODY_INSTRUCTION_DESCRIPTION)
    evidence_ids: tuple[str, ...] = ()


class WireCopyAnswerBody(BodyModel):
    """Reuse a displayed answer including its existing sources and limitations.

    TXT serialization and inclusion of existing citations are server export
    behavior, not transformations of the answer's wording or content. A newly
    requested displayed layout uses transform_answer even if wording stays literal.
    """
    kind: Literal["copy_answer"] = "copy_answer"
    source: AnswerReference | OrdinalAnswerReference


class CopyAnswerBody(BodyModel):
    kind: Literal["copy_answer"] = "copy_answer"
    source: BoundAnswerReference


class WireTransformAnswerBody(BodyModel):
    """Change the actual answer body, for example its language, length or claims.

    Saving the existing answer with its sources/limitations is copy_answer.
    """
    kind: Literal["transform_answer"] = "transform_answer"
    source: AnswerReference | OrdinalAnswerReference
    instruction: str = Field(min_length=1, description=(
        "The requested change to the answer's actual wording, language, structure or claims. "
        "TXT serialization or including existing sources/limitations is not a body change; use copy_answer. "
        + BODY_INSTRUCTION_DESCRIPTION
    ))
    evidence_ids: tuple[str, ...] = ()


class TransformAnswerBody(BodyModel):
    kind: Literal["transform_answer"] = "transform_answer"
    source: BoundAnswerReference
    instruction: str = Field(min_length=1, description=BODY_INSTRUCTION_DESCRIPTION)
    evidence_ids: tuple[str, ...] = ()


class ExtractBody(BodyModel):
    kind: Literal["extract"] = "extract"
    instruction: str = Field(default="", description=BODY_INSTRUCTION_DESCRIPTION)
    requirement_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


class UnresolvedBody(BodyModel):
    kind: Literal["unresolved"] = "unresolved"
    question: str = Field(min_length=1)
    instruction: str = Field(default="", description=BODY_INSTRUCTION_DESCRIPTION)


# Unique Literal tags make branches disjoint. A plain union emits nested anyOf
# for the provider rather than relying on discriminator/oneOf dialect support.
WireBodyPlan = Union[ComposeBody, AcknowledgeBody, WireCopyInputBody, WireTransformInputBody,
                     WireCopyAnswerBody, WireTransformAnswerBody, ExtractBody, UnresolvedBody]
BodyPlan = Union[ComposeBody, AcknowledgeBody, CopyInputBody, TransformInputBody,
                 CopyAnswerBody, TransformAnswerBody, ExtractBody, UnresolvedBody]


def body_to_wire(body: BodyPlan) -> WireBodyPlan:
    if isinstance(body, (CopyInputBody, TransformInputBody)):
        selector = InputTextReference(turn_id=body.source.turn_id, quote=body.source.text)
        if isinstance(body, CopyInputBody):
            return WireCopyInputBody(source=selector)
        return WireTransformInputBody(source=selector, instruction=body.instruction, evidence_ids=body.evidence_ids)
    if isinstance(body, (CopyAnswerBody, TransformAnswerBody)):
        selector = AnswerReference(ref=body.source.ref)
        if isinstance(body, CopyAnswerBody):
            return WireCopyAnswerBody(source=selector)
        return WireTransformAnswerBody(source=selector, instruction=body.instruction, evidence_ids=body.evidence_ids)
    return body
