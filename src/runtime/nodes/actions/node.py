from __future__ import annotations

import logging
from typing import Any

from src.core.answer_schema import ActionReceipt, AnswerResponse, export_answer_text
from src.core.contracts import GraphState, SlackDestination
from src.core.contracts.graph_state import PendingAction
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.message_utils import build_tool_message
from src.core.request_contracts import check_answer_contract, resolve_body_response
from src.infra.logging_utils import log_event
from src.runtime.nodes.actions.policy import get_slack_destinations
from src.runtime.nodes.actions.receipts import build_save_receipt, build_slack_receipt


logger = logging.getLogger(__name__)


def _ready_body(*, contract, response, runtime, pending, planner) -> bool:
    """Only this contract revision's checked complete body can leave the app."""
    if (
        not contract.can_prepare_body() or response.kind != "answer"
        or response.request_id != contract.request_id
        or response.contract_revision != contract.revision
        or planner.diagnostics.reason in {"upload_retriever_missing", "planner_unavailable"}
        or not response.result.content.blocks
        or not check_answer_contract(
            contract.answer, response.result.content,
            evidence=[citation.evidence for citation in response.result.citations],
        ).valid
    ):
        return False
    if contract.body.kind == "copy_input":
        return export_answer_text(response.result) == contract.body.source.text
    if contract.body.kind in {"copy_answer", "transform_answer"}:
        source = resolve_body_response(
            contract, previous_response=runtime.previous_response,
            pending_response=pending.response if pending else None,
        )
        if source is None:
            return False
        if contract.body.kind == "copy_answer" and response.result.content_hash != source.content_hash:
            return False
    return True


def _action_question(contract, action: str) -> str:
    slot = "save_intent" if action == "save_text" else "slack_intent"
    question = next((item.question for item in contract.missing_info if item.slot == slot), None)
    return question or ("이 본문을 파일로 저장할까요?" if action == "save_text" else "이 본문을 Slack으로 보낼까요?")


def _pending_response(*, contract, runtime, pending, response, body_ready: bool) -> AnswerResponse | None:
    if body_ready:
        return response.result.model_copy(update={"actions": []}, deep=True)
    if pending is not None and pending.contract.request_id == contract.request_id:
        return pending.response.model_copy(deep=True) if pending.response is not None else None
    source = resolve_body_response(
        contract, previous_response=runtime.previous_response,
        pending_response=pending.response if pending else None,
    )
    return source.model_copy(update={"actions": []}, deep=True) if source is not None else None


def _prepared_pending_body(*, contract, pending, body_ready: bool) -> bool:
    if body_ready:
        return True
    return bool(
        pending is not None and pending.body_prepared and pending.response is not None
        and pending.contract.request_id == contract.request_id
        and contract.can_prepare_body() and contract.body.kind == "copy_answer"
        and contract.body.source.ref == "pending"
        and contract.body.source.response_hash == pending.response.content_hash
        and check_answer_contract(contract.answer, pending.response.content,
                                  evidence=[citation.evidence for citation in pending.response.citations]).valid
    )


def make_action_postprocess_node(
    save_text_tool: Any, slack_notify_tool: Any, verbose: bool,
    has_default_slack_destination: bool = False,
):
    def action_postprocess(state: GraphState) -> GraphState:
        runtime = get_runtime_state(state)
        contract = runtime.request_contract
        if contract is None or contract.failure is not None:
            return {}
        pending = runtime.pending_action
        if any(item.slot == "pending_request" for item in contract.missing_info):
            return {}
        if contract.target_request_id is not None and (
            pending is None or contract.target_request_id != pending.contract.request_id
        ):
            return {}
        if contract.relation in {"supplement", "cancel"} and contract.target_request_id is None:
            return {}
        updates: GraphState = {}
        cancellation_resolved = contract.can_cancel_pending()
        if contract.can_acknowledge() and contract.relation != "cancel":
            if (pending is not None and contract.relation in {"correction", "supplement"}
                    and contract.target_request_id == pending.contract.request_id):
                updates["runtime"] = runtime.model_copy(update={"pending_action": None})
            return updates
        if (contract.relation == "new" or cancellation_resolved) and pending is not None:
            updates["runtime"] = runtime.model_copy(update={"pending_action": None})
        if cancellation_resolved:
            return updates
        if contract.relation == "cancel":
            if pending is not None:
                updates["runtime"] = runtime.model_copy(update={"pending_action": pending.model_copy(
                    update={"contract": contract, "phase": "awaiting_input"}, deep=True,
                )})
            return updates
        planner = get_planner_state(state)
        response = get_response_state(state)
        if contract.relation == "supplement" and pending is None:
            return updates
        body_ready = _ready_body(contract=contract, response=response, runtime=runtime, pending=pending, planner=planner)
        intents = {name: getattr(contract.actions, name).intent for name in ("save_text", "slack_notify")}
        waiting_intents = {
            name for name, slot in (("save_text", "save_intent"), ("slack_notify", "slack_intent"))
            if intents[name] == "unresolved" or any(item.slot == slot for item in contract.missing_info)
        }
        if not any(intent == "requested" for intent in intents.values()) and not waiting_intents:
            if not body_ready:
                completed = pending.completed_actions if pending is not None and pending.contract.request_id == contract.request_id else ()
                updates["runtime"] = runtime.model_copy(update={"pending_action": PendingAction(
                    contract=contract,
                    response=_pending_response(contract=contract, runtime=runtime, pending=pending,
                                               response=response, body_ready=False),
                    phase="awaiting_body" if contract.can_prepare_body() else "awaiting_input",
                    completed_actions=completed, body_prepared=False,
                )})
            elif pending is not None and contract.relation in {"supplement", "correction"}:
                updates["runtime"] = runtime.model_copy(update={"pending_action": None})
            return updates
        debug = get_debug_state(state)
        body = export_answer_text(response.result, include_sources=True) if body_ready else ""
        destinations = (
            SlackDestination.model_validate(contract.slack_destination.model_dump())
            if contract.slack_destination is not None
            else get_slack_destinations(runtime.session_metadata)
        )
        slack_available = (destinations.has_destination() or has_default_slack_destination) and not any(
            item.slot == "slack_destination" and item.reason != "not_provided" for item in contract.missing_info
        )
        completed = set(pending.completed_actions) if pending and pending.contract.request_id == contract.request_id else set()
        receipts: list[ActionReceipt] = []
        messages = []
        action_errors: list[str] = []
        error_codes = []
        needs_destination = False
        needs_intent = bool(waiting_intents)

        if intents["save_text"] == "requested" and "save_text" not in completed:
            if contract.execution_ready("save_text", body_ready=body_ready):
                try:
                    result = save_text_tool(content=body, filename_prefix="response")
                except Exception as exc:
                    result = {"status": "error", "error": str(exc)}
                messages.append(build_tool_message("save_text", result, 1))
                receipt = build_save_receipt(result)
                receipts.append(receipt)
                if receipt.status == "success":
                    completed.add("save_text")

        if intents["slack_notify"] == "requested" and "slack_notify" not in completed:
            if (body_ready and "slack_notify" not in waiting_intents
                    and not contract.execution_ready("slack_notify", body_ready=True, destination_ready=slack_available)):
                needs_destination = True
                question = next((item.question for item in contract.missing_info if item.slot == "slack_destination"), None)
                receipts.append(ActionReceipt(
                    kind="slack_notify", status="skipped",
                    message=question or "Slack으로 보낼 대상의 channel_id, user_id 또는 email을 알려주세요.",
                ))
            elif contract.execution_ready("slack_notify", body_ready=body_ready, destination_ready=slack_available):
                try:
                    result = slack_notify_tool(
                        text=body, user_id=destinations.user_id,
                        email=destinations.email, channel_id=destinations.channel_id,
                    )
                except Exception as exc:
                    result = {"status": "error", "error": str(exc), "error_code": "SLACK_AUTH_FAILED"}
                messages.append(build_tool_message("slack_notify", result, 1))
                receipt = build_slack_receipt(slack_result=result, destinations=destinations)
                receipts.append(receipt)
                if receipt.status == "success":
                    completed.add("slack_notify")
                if isinstance(result, dict) and result.get("error_code") == "SLACK_DESTINATION_MISSING":
                    needs_destination = True
                if isinstance(result, dict) and result.get("error_code"):
                    error_codes.append(result["error_code"])

        if body_ready:
            receipts.extend(ActionReceipt(kind=name, status="skipped", message=_action_question(contract, name))
                            for name in intents if name in waiting_intents)

        for receipt in receipts:
            if receipt.status == "error":
                action_errors.append(f"{receipt.kind}: {receipt.error}")

        if receipts:
            updates["response"] = response.model_copy(update={
                "result": response.result.model_copy(update={"actions": [*response.result.actions, *receipts]}),
            })
        waiting_for_body = not body_ready and any(intent == "requested" for intent in intents.values())
        waiting_for_delivery = bool(action_errors)
        phase = ("awaiting_input" if needs_intent or not contract.can_prepare_body() else
                 "awaiting_body" if waiting_for_body else
                 "awaiting_delivery" if waiting_for_delivery else "awaiting_destination")
        next_pending = (
            PendingAction(
                contract=contract,
                response=_pending_response(contract=contract, runtime=runtime, pending=pending,
                                           response=response, body_ready=body_ready),
                completed_actions=tuple(sorted(completed)),
                phase=phase,
                body_prepared=_prepared_pending_body(contract=contract, pending=pending, body_ready=body_ready),
            )
            if needs_destination or needs_intent or waiting_for_body or waiting_for_delivery else None
        )
        if next_pending is not None or pending is not None:
            updates["runtime"] = runtime.model_copy(update={"pending_action": next_pending})
        if messages:
            updates["messages"] = messages
        if action_errors:
            updates["debug"] = debug.model_copy(update={
                "action_errors": [*debug.action_errors, *action_errors],
                "error_codes": list(dict.fromkeys([*debug.error_codes, *error_codes])),
            })
        if verbose and messages:
            log_event(logger, logging.INFO, "postprocess", tools=", ".join(message.name or "" for message in messages))
        return updates

    return action_postprocess
