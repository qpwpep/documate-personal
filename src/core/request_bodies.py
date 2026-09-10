"""Body requests distinguish model-selected references from server-bound data."""
from __future__ import annotations

import hashlib
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class BoundAnswerReference(BodyModel):
    ref: Literal["previous", "pending"]
    response_hash: str = Field(min_length=1)


class ComposeBody(BodyModel):
    kind: Literal["compose"] = "compose"
    instruction: str = ""
    evidence_ids: tuple[str, ...] = ()


class AcknowledgeBody(BodyModel):
    """Confirm a complete instruction without creating a deliverable document."""
    kind: Literal["acknowledge"] = "acknowledge"
    evidence_ids: tuple[str, ...] = ()


class WireCopyInputBody(BodyModel):
    kind: Literal["copy_input"] = "copy_input"
    source: InputTextReference


class CopyInputBody(BodyModel):
    kind: Literal["copy_input"] = "copy_input"
    source: BoundInputText


class WireTransformInputBody(BodyModel):
    kind: Literal["transform_input"] = "transform_input"
    source: InputTextReference
    instruction: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()


class TransformInputBody(BodyModel):
    kind: Literal["transform_input"] = "transform_input"
    source: BoundInputText
    instruction: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()


class WireCopyAnswerBody(BodyModel):
    kind: Literal["copy_answer"] = "copy_answer"
    source: AnswerReference


class CopyAnswerBody(BodyModel):
    kind: Literal["copy_answer"] = "copy_answer"
    source: BoundAnswerReference


class WireTransformAnswerBody(BodyModel):
    kind: Literal["transform_answer"] = "transform_answer"
    source: AnswerReference
    instruction: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()


class TransformAnswerBody(BodyModel):
    kind: Literal["transform_answer"] = "transform_answer"
    source: BoundAnswerReference
    instruction: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()


class ExtractBody(BodyModel):
    kind: Literal["extract"] = "extract"
    instruction: str = ""
    requirement_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


class UnresolvedBody(BodyModel):
    kind: Literal["unresolved"] = "unresolved"
    question: str = Field(min_length=1)
    instruction: str = ""


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
