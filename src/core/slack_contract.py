"""Recipient identity and delivery facts shared by every Slack consumer."""
from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SlackModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecipientSelector(SlackModel):
    kind: Literal["channel", "user", "email"]
    value: str = Field(min_length=1, strict=True)

    @field_validator("value")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("an explicit Slack recipient must not be blank")
        return value

    @model_validator(mode="after")
    def valid_selector(self) -> "RecipientSelector":
        pattern = {"channel": r"[CGD][A-Z0-9]+", "user": r"[UW][A-Z0-9]+",
                   "email": r"[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+"}[self.kind]
        if not re.fullmatch(pattern, self.value):
            raise ValueError(f"invalid Slack {self.kind} recipient")
        return self


class OmittedRecipient(SlackModel):
    state: Literal["omitted"] = "omitted"


class ExplicitRecipient(SlackModel):
    state: Literal["explicit"] = "explicit"
    selector: RecipientSelector
    evidence_ids: tuple[str, ...] = ()


class UnresolvedRecipient(SlackModel):
    state: Literal["unresolved"] = "unresolved"
    raw_input: str
    reason: Literal["invalid", "ambiguous", "unverified", "conflict"]
    evidence_ids: tuple[str, ...] = ()


RecipientIntent = Annotated[
    OmittedRecipient | ExplicitRecipient | UnresolvedRecipient, Field(discriminator="state")
]


class RecipientSelection(SlackModel):
    request_id: str = Field(min_length=1)
    source: Literal["user_text", "request_input", "configured_default"]
    selector: RecipientSelector


SlackErrorCode = Literal[
    "recipient_missing", "recipient_invalid", "recipient_ambiguous", "recipient_conflict",
    "configuration_error", "target_not_found", "target_unavailable", "permission_denied",
    "authentication_failed", "rate_limited", "temporary_failure", "delivery_unknown",
    "protocol_error",
]


class SlackFailure(SlackModel):
    stage: Literal["input", "selection", "lookup", "open_dm", "send"]
    code: SlackErrorCode
    message: str = Field(min_length=1)
    next_action: Literal["correct_input", "fix_configuration", "retry_same_target", "verify_delivery"]
    retry_after_seconds: int | None = Field(default=None, ge=0)
    slack_error: str | None = None


class SlackDefault(SlackModel):
    selector: RecipientSelector | None = None
    failure: SlackFailure | None = None

    @model_validator(mode="after")
    def single_default(self) -> "SlackDefault":
        if self.selector is not None and self.failure is not None:
            raise ValueError("a default recipient cannot be valid and invalid at once")
        return self


class SlackTarget(SlackModel):
    channel_id: str
    user_id: str | None = None

    @model_validator(mode="after")
    def addressed_target(self) -> "SlackTarget":
        RecipientSelector(kind="channel", value=self.channel_id)
        if self.user_id is not None:
            RecipientSelector(kind="user", value=self.user_id)
        return self


class SlackDelivery(SlackModel):
    """The selected identity survives partial resolution and later retries.

    Optional IDs mean resolution has not reached that stage. Failure is always
    explicit in status/failure, never represented by an absent ID alone.
    """

    invocation_id: str | None = Field(default=None, min_length=1)
    intent: RecipientIntent
    selection: RecipientSelection | None = None
    resolved_user_id: str | None = None
    target: SlackTarget | None = None
    status: Literal["pending", "sent", "not_sent", "unknown"] = "pending"
    failure: SlackFailure | None = None
    message_ts: str | None = None

    @model_validator(mode="after")
    def preserve_identity(self) -> "SlackDelivery":
        selection = self.selection
        if selection is not None:
            if self.intent.state == "unresolved":
                raise ValueError("an unresolved recipient cannot have a selected target")
            if self.intent.state == "explicit" and self.intent.selector != selection.selector:
                raise ValueError("selection must preserve the explicit recipient")
            if selection.source == "configured_default" and self.intent.state != "omitted":
                raise ValueError("only an omitted recipient can use the configured default")
            if self.intent.state == "omitted" and selection.source != "configured_default":
                raise ValueError("explicit selections must retain their recipient intent")
        if self.resolved_user_id is not None:
            RecipientSelector(kind="user", value=self.resolved_user_id)
            if selection is None or selection.selector.kind == "channel":
                raise ValueError("resolved user requires a selected user or email")
            if selection.selector.kind == "user" and selection.selector.value != self.resolved_user_id:
                raise ValueError("resolved user differs from the selected user")
        if self.target is not None:
            if selection is None:
                raise ValueError("a send target requires a recipient selection")
            selector = selection.selector
            if selector.kind == "channel":
                if self.target.channel_id != selector.value or self.target.user_id is not None:
                    raise ValueError("send channel differs from the selected channel")
            elif (not self.resolved_user_id or self.target.user_id != self.resolved_user_id
                  or not self.target.channel_id.startswith("D")):
                raise ValueError("DM target must belong to the resolved user")
        if self.status == "sent":
            if self.target is None or not self.message_ts or not self.message_ts.strip() or self.failure is not None:
                raise ValueError("sent delivery requires a confirmed target and message timestamp")
        elif self.message_ts is not None:
            raise ValueError("only confirmed delivery has a message timestamp")
        if self.status in {"not_sent", "unknown"} and self.failure is None:
            raise ValueError("unsuccessful delivery requires a structured failure")
        if self.status == "pending" and self.failure is not None:
            raise ValueError("pending delivery cannot hide a previous failure")
        if self.status == "unknown" and (
            self.target is None or self.failure.stage != "send"
            or self.failure.next_action != "verify_delivery"
        ):
            raise ValueError("unknown delivery requires an attempted send and delivery verification")
        if self.failure is not None and self.status != "unknown" and (
            self.failure.code == "delivery_unknown" or self.failure.next_action == "verify_delivery"
        ):
            raise ValueError("delivery uncertainty cannot be represented as a safe failure")
        return self
