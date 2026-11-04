"""An unfinished request owns its body and remaining delivery obligations."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.core.answer_export import answer_export_key
from src.core.answer_schema import ActionReceipt, AnswerResponse
from src.core.request_contracts import (
    AnswerContract, CopyAnswerBody, RequestContract, UnresolvedBody, check_answer_contract, resolve_body_response,
)
from src.core.save_contract import SaveOperation
from src.core.slack_contract import SlackDelivery


class PendingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: RequestContract
    response: AnswerResponse | None = None
    body_prepared: bool = False
    phase: Literal["awaiting_input", "awaiting_body", "awaiting_destination", "awaiting_delivery"] = "awaiting_destination"
    completed_actions: tuple[Literal["save_text", "slack_notify"], ...] = ()
    save_operation: SaveOperation | None = None
    save_receipt: ActionReceipt | None = None
    slack_delivery: SlackDelivery | None = None

    @classmethod
    def after_execution(
        cls, *, contract: RequestContract, previous_response: AnswerResponse | None,
        pending: PendingRequest | None, response: AnswerResponse, body_ready: bool,
        completed_actions: tuple[Literal["save_text", "slack_notify"], ...],
        save_operation: SaveOperation | None, save_receipt: ActionReceipt | None,
        slack_delivery: SlackDelivery | None, receipts: list[ActionReceipt], needs_destination: bool = False,
    ) -> PendingRequest | None:
        """Retain precisely the unfinished body and delivery obligations after execution."""
        needs_intent = any(
            getattr(contract.actions, name).intent == "unresolved"
            or any(item.slot == slot for item in contract.missing_info)
            for name, slot in (("save_text", "save_intent"), ("slack_notify", "slack_intent"))
        )
        waiting_for_delivery = any(receipt.status in {"error", "unknown"} for receipt in receipts) or bool(
            contract.action_requested("slack_notify") and slack_delivery is not None
            and slack_delivery.status != "sent" and not needs_destination and body_ready
        )
        if not (needs_destination or needs_intent or not body_ready or waiting_for_delivery):
            return None
        if needs_intent or not contract.can_prepare_body():
            phase = "awaiting_input"
        elif not body_ready:
            phase = "awaiting_body"
        elif waiting_for_delivery:
            phase = "awaiting_delivery"
        else:
            phase = "awaiting_destination"
        if body_ready:
            retained = response.model_copy(update={"actions": []}, deep=True)
        elif pending is not None and pending.contract.request_id == contract.request_id:
            retained = pending.response.model_copy(deep=True) if pending.response is not None else None
        else:
            source = resolve_body_response(
                contract, previous_response=previous_response,
                pending_response=pending.response if pending is not None else None,
            )
            retained = source.model_copy(update={"actions": []}, deep=True) if source is not None else None
        has_delivery = needs_intent or any(contract.action_requested(name) for name in ("save_text", "slack_notify"))
        prepared = body_ready or bool(
            has_delivery and pending is not None and pending.body_prepared and pending.response is not None
            and pending.contract.request_id == contract.request_id and contract.can_prepare_body()
            and contract.body.kind == "copy_answer" and contract.body.source.ref == "pending"
            and contract.body.source.response_hash == pending.response.content_hash
            and _answer_conditions_ready(contract, pending.response)
        )
        return cls(
            contract=contract, response=retained, body_prepared=prepared, phase=phase,
            completed_actions=completed_actions, save_operation=save_operation,
            save_receipt=save_receipt, slack_delivery=slack_delivery,
        )

    def reconcile(self, contract: RequestContract, *, recipient_confirmed: bool) -> PendingRequest:
        """Accept one grounded correction while preserving completed or unknown effects."""
        completed = tuple(name for name in self.completed_actions
                          if getattr(contract.actions, name).intent != "requested")
        body_changed = self.body_changed(contract)
        delivery = self.slack_delivery
        recipient_changed = (contract.slack_recipient.model_dump(exclude={"evidence_ids"})
                             != self.contract.slack_recipient.model_dump(exclude={"evidence_ids"}))
        if recipient_changed and (delivery is None or delivery.status != "unknown"):
            delivery = None
        elif (delivery is not None and delivery.selection is None
              and delivery.intent.state == "unresolved" and contract.slack_recipient.state == "explicit"
              and recipient_confirmed):
            delivery = None
        elif ("slack_notify" in self.completed_actions and contract.action_requested("slack_notify")
              and delivery is not None and delivery.status == "sent"):
            # New authorization creates a new obligation but keeps the selected identity.
            delivery = delivery.model_copy(update={
                "status": "pending", "failure": None, "message_ts": None,
                "operation": None, "invocation_id": None,
            })
        if body_changed and delivery is not None and delivery.status in {"pending", "not_sent"}:
            delivery = delivery.model_copy(update={"operation": None, "invocation_id": None})
        new_save = body_changed or ("save_text" in self.completed_actions and contract.action_requested("save_text"))
        reset_body = body_changed or contract.body.kind != "copy_answer"
        body_prepared = False if reset_body else self.body_prepared
        phase = self.phase
        if reset_body:
            phase = "awaiting_body" if contract.can_prepare_body() else "awaiting_input"
        return self.model_copy(update={
            "contract": contract, "completed_actions": completed, "slack_delivery": delivery,
            "save_operation": None if new_save else self.save_operation,
            "save_receipt": None if new_save else self.save_receipt,
            "body_prepared": body_prepared, "phase": phase,
        })

    def body_changed(self, contract: RequestContract) -> bool:
        """Compare accepted facts rather than newly bound evidence IDs or source labels."""
        if _answer_facts(contract.answer) != _answer_facts(self.contract.answer):
            return True
        if (isinstance(contract.body, CopyAnswerBody) and self.response is not None
                and contract.body.source.response_hash == self.response.content_hash):
            return False
        body, previous = contract.body, self.contract.body
        if isinstance(body, UnresolvedBody):
            body = contract.body_request or body
            previous = self.contract.body_request or previous
        exclude = {"evidence_ids": True, "source": {"ref"}}
        return body.model_dump(exclude=exclude) != previous.model_dump(exclude=exclude)

    def owns_save(self, answer: AnswerResponse) -> bool:
        return bool(self.body_prepared and self.response is not None
                    and self.contract.action_requested("save_text")
                    and answer_export_key(self.response) == answer_export_key(answer))

    def complete_save(
        self, answer: AnswerResponse, operation: SaveOperation, receipt: ActionReceipt,
    ) -> PendingRequest | None:
        """An explicit export can fulfill the same prepared request's remaining save."""
        if receipt.status != "success" or not self.owns_save(answer):
            return self
        completed = set(self.completed_actions) | {"save_text"}
        waiting_slack = (self.contract.actions.slack_notify.intent in {"requested", "unresolved"}
                         and "slack_notify" not in completed)
        if not waiting_slack:
            return None
        delivery = self.slack_delivery
        needs_destination = delivery is None or delivery.selection is None or bool(
            delivery.failure is not None and delivery.failure.next_action == "correct_input"
        )
        assert self.response is not None
        return self.after_execution(
            contract=self.contract, previous_response=None, pending=self, response=self.response,
            body_ready=True, completed_actions=tuple(sorted(completed)), save_operation=operation,
            save_receipt=receipt, slack_delivery=delivery, receipts=[receipt], needs_destination=needs_destination,
        )


def _answer_facts(answer: AnswerContract) -> dict[str, object]:
    return {
        "content": tuple(item.model_dump(exclude={"evidence_ids"}) for item in answer.content),
        "format": tuple(item.model_dump(exclude={"evidence_ids"}) for item in answer.format),
        "sentence_counts": tuple(item.model_dump(exclude={"evidence_ids"}) for item in answer.sentence_counts),
        "citation_scope": (answer.citation_scope.model_dump(exclude={"evidence_ids"})
                           if answer.citation_scope is not None else None),
        "preferences": answer.preferences,
    }


def _answer_conditions_ready(contract: RequestContract, answer: AnswerResponse) -> bool:
    check = check_answer_contract(
        contract.answer, answer.content, evidence=[citation.evidence for citation in answer.citations],
    )
    if not check.valid:
        return False
    if not check.unchecked_semantic:
        return True
    try:
        report = AnswerResponse.model_validate(answer.model_dump(mode="json")).semantic_validation
    except ValueError:
        return False
    return report is not None and report.passed and any(
        item.name == "request_fulfilled" and item.unit_id is None for item in report.checks
    )
