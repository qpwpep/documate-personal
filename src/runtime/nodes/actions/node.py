from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from src.core.answer_schema import ActionReceipt, export_answer_text
from src.core.answer_export import answer_export_key, answer_export_snapshot
from src.core.contracts import GraphState
from src.core.pending_request import PendingRequest
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.message_utils import build_tool_message
from src.core.llm_errors import LLMCallError
from src.core.request_contracts import resolve_body_response
from src.core.save_contract import SaveOperation, SaveToolResult
from src.infra.logging_utils import log_event
from src.infra.llm_boundary import check_request_deadline
from src.infra.answer_export import build_save_failure, build_save_receipt, execute_save, select_answer_save_operation
from src.infra.saved_artifacts import ArtifactError
from src.runtime.nodes.actions.policy import select_slack_delivery
from src.core.slack_contract import SlackDefault, SlackDelivery
from src.runtime.nodes.actions.receipts import build_slack_receipt
from src.runtime.agent_runtime.tool_execution import invoke_tool, record_nonexecution


logger = logging.getLogger(__name__)


def _save_operation(*, contract, runtime, pending, response, body: str) -> SaveOperation:
    prior = pending.save_operation if pending and pending.contract.request_id == contract.request_id else None
    scope = runtime.session_id or contract.request_id
    if contract.body.kind == "copy_answer":
        return select_answer_save_operation(
            response.result, session_id=scope,
            remembered=runtime.answer_export_operations.get(answer_export_key(response.result)), pending=prior,
        )
    if prior is not None and prior.request_id == contract.request_id and prior.session_id == scope:
        # Request reconciliation already discarded replaced obligations. A
        # surviving operation stays frozen across retries and destination edits.
        return prior
    source = getattr(contract.body, "source", None)
    source_hash = (getattr(source, "response_hash", None) or getattr(source, "content_hash", None)
                   or response.result.content_hash)
    return SaveOperation.for_text(
        body, operation_id=uuid5(NAMESPACE_URL, f"save:{contract.request_id}:{contract.revision}").hex,
        session_id=scope,
        request_id=contract.request_id, contract_revision=contract.revision,
        target_kind=contract.body.kind, source_hash=source_hash, answer_hash=response.result.content_hash,
    )


def _ready_body(*, contract, response, runtime, pending, planner) -> bool:
    """Only this contract revision's checked complete body can leave the app."""
    if (
        not contract.can_prepare_body() or response.kind != "answer"
        or response.request_id != contract.request_id
        or response.contract_revision != contract.revision
        or planner.diagnostics.reason == "upload_retriever_missing"
        or not response.result.content.blocks
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
        if contract.body.kind == "copy_answer":
            copied = answer_export_snapshot(response.result).model_dump(exclude={"semantic_validation"})
            original = answer_export_snapshot(source).model_dump(exclude={"semantic_validation"})
            if copied != original:
                return False
    return True


def _action_question(contract, action: str) -> str:
    slot = "save_intent" if action == "save_text" else "slack_intent"
    question = next((item.question for item in contract.missing_info if item.slot == slot), None)
    return question or ("이 본문을 파일로 저장할까요?" if action == "save_text" else "이 본문을 Slack으로 보낼까요?")


def make_action_postprocess_node(
    save_text_tool: Callable[..., SaveToolResult], slack_notify_tool: Callable[..., SlackDelivery], verbose: bool,
    default_slack_recipient: SlackDefault | None = None,
):
    default_slack_recipient = default_slack_recipient or SlackDefault()

    def action_postprocess(state: GraphState) -> GraphState:
        runtime = get_runtime_state(state)
        contract = runtime.request_contract
        if contract is None:
            return {}
        pending = runtime.pending_request
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
                updates["runtime"] = runtime.model_copy(update={"pending_request": None})
            return updates
        if (contract.relation == "new" or cancellation_resolved) and pending is not None:
            updates["runtime"] = runtime.model_copy(update={"pending_request": None})
        if cancellation_resolved:
            return updates
        if contract.relation == "cancel":
            if pending is not None:
                updates["runtime"] = runtime.model_copy(update={"pending_request": pending.model_copy(
                    update={"contract": contract, "phase": "awaiting_input"}, deep=True,
                )})
            return updates
        planner = get_planner_state(state)
        response = get_response_state(state)
        if contract.relation == "supplement" and pending is None:
            return updates
        body_ready = _ready_body(contract=contract, response=response, runtime=runtime, pending=pending, planner=planner)
        intents = {name: getattr(contract.actions, name).intent for name in ("save_text", "slack_notify")}
        for name, intent in intents.items():
            if intent in {"forbidden", "unresolved"}:
                record_nonexecution(name, reason_code=f"intent_{intent}")
        waiting_intents = {
            name for name, slot in (("save_text", "save_intent"), ("slack_notify", "slack_intent"))
            if intents[name] == "unresolved" or any(item.slot == slot for item in contract.missing_info)
        }
        if not any(intent == "requested" for intent in intents.values()) and not waiting_intents:
            if not body_ready:
                completed = pending.completed_actions if pending is not None and pending.contract.request_id == contract.request_id else ()
                same_request = pending is not None and pending.contract.request_id == contract.request_id
                updates["runtime"] = runtime.model_copy(update={"pending_request": PendingRequest.after_execution(
                    contract=contract, previous_response=runtime.previous_response, pending=pending,
                    response=response.result, body_ready=False, completed_actions=completed,
                    save_operation=pending.save_operation if same_request else None,
                    save_receipt=pending.save_receipt if same_request else None,
                    slack_delivery=pending.slack_delivery if same_request else None, receipts=[],
                )})
            elif pending is not None and contract.relation in {"supplement", "correction"}:
                updates["runtime"] = runtime.model_copy(update={"pending_request": None})
            return updates
        debug = get_debug_state(state)
        body = export_answer_text(response.result, include_sources=True) if body_ready else ""
        slack_delivery = pending.slack_delivery if pending and pending.contract.request_id == contract.request_id else None
        if intents["slack_notify"] == "requested":
            slack_delivery = select_slack_delivery(
                request_id=contract.request_id, intent=contract.slack_recipient,
                request_recipient=runtime.session_metadata.slack_recipient,
                default=default_slack_recipient, previous=slack_delivery,
            )
        completed = set(pending.completed_actions) if pending and pending.contract.request_id == contract.request_id else set()
        save_operation = pending.save_operation if pending and pending.contract.request_id == contract.request_id else None
        save_receipt = pending.save_receipt if pending and pending.contract.request_id == contract.request_id else None
        receipts: list[ActionReceipt] = []
        messages = []
        action_errors: list[str] = []
        error_codes = []
        needs_destination = False

        if intents["save_text"] == "requested" and "save_text" not in completed:
            if contract.execution_ready("save_text", body_ready=body_ready):
                invocation_id = None
                try:
                    check_request_deadline("action_postprocess")
                    save_operation = _save_operation(contract=contract, runtime=runtime, pending=pending,
                                                     response=response, body=body)
                    if save_operation.answer_hash != response.result.content_hash:
                        invocation_id = record_nonexecution("save_text", reason_code="idempotency_conflict")
                        raise ArtifactError("idempotency_conflict", "재시도 본문이 확정된 저장 대상과 다릅니다.")
                    invocation_id = uuid4().hex
                    check_request_deadline("action_postprocess")
                    receipt = execute_save(
                        lambda: invoke_tool("save_text", save_text_tool, invocation_id=invocation_id,
                                            content=body, operation=save_operation),
                        operation=save_operation, invocation_id=invocation_id,
                    )
                except LLMCallError:
                    # A request deadline is not a failed filesystem write.
                    raise
                except Exception as exc:
                    receipt = build_save_failure(
                        exc.code if isinstance(exc, ArtifactError) else "write_failed", str(exc),
                        operation=save_operation, invocation_id=invocation_id,
                    )
                save_receipt = receipt
                messages.append(build_tool_message("save_text", {
                    **receipt.model_dump(mode="json", exclude_none=True),
                    "bytes": receipt.artifact.byte_count if receipt.artifact else 0,
                }, 1))
                receipts.append(receipt)
                if receipt.status == "success":
                    completed.add("save_text")
            else:
                record_nonexecution("save_text", reason_code="execution_not_ready")
        elif intents["save_text"] == "requested" and save_receipt is not None and save_operation is not None:
            # Re-entering an unchanged active obligation may reuse its receipt.
            # The planner normally removes completed actions on destination-only
            # follow-ups; this branch must never attach old success to new text.
            if (body_ready and save_operation.answer_hash == response.result.content_hash
                    and save_operation.matches_bytes(body.encode("utf-8-sig"))):
                result = SaveToolResult(file_path=Path(save_receipt.file_path),
                                        operation=save_receipt.operation, artifact=save_receipt.artifact)
                receipt = build_save_receipt(result, operation=save_operation, invocation_id=save_receipt.invocation_id)
            else:
                receipt = build_save_failure("idempotency_conflict", "현재 답변이 확정된 저장 대상과 다릅니다.",
                                             operation=save_operation)
            if receipt.status == "success":
                record_nonexecution("save_text", phase="reused", reason_code="completed_operation",
                                    origin_invocation_id=save_receipt.invocation_id)
            else:
                # A rejected reuse is a new pre-execution decision. It cannot
                # relabel the original successful invocation as a failure.
                invocation_id = record_nonexecution(
                    "save_text", reason_code=receipt.error_code or "reuse_unverifiable",
                )
                receipt = receipt.model_copy(update={"invocation_id": invocation_id})
            save_receipt = receipt
            receipts.append(receipt)
            if receipt.status != "success":
                completed.discard("save_text")

        if intents["slack_notify"] == "requested" and "slack_notify" not in completed:
            if (body_ready and slack_delivery.status == "unknown"
                    and (not slack_delivery.operation.matches_text(body)
                         or (contract.slack_recipient.state == "explicit"
                             and contract.slack_recipient.selector != slack_delivery.selection.selector))):
                invocation_id = record_nonexecution("slack_notify", reason_code="previous_delivery_unknown")
                unchanged_body = slack_delivery.operation.matches_text(body)
                receipts.append(ActionReceipt(
                    kind="slack_notify", status="skipped", invocation_id=invocation_id,
                    previous_slack=slack_delivery,
                    message=("변경한 수신자에게 전송하지 않았습니다. " if unchanged_body else "수정된 본문은 전송하지 않았습니다. ")
                    + "이전 본문의 Slack 전달 여부를 먼저 확인한 뒤, 추가 전송이 필요하면 새 전송 요청을 해 주세요.",
                ))
            elif body_ready and "slack_notify" not in waiting_intents:
                if slack_delivery.selection is not None and contract.execution_ready(
                    "slack_notify", body_ready=True, destination_ready=True,
                ):
                    if slack_delivery.status in {"sent", "unknown"}:
                        record_nonexecution("slack_notify", phase="reused", reason_code="existing_delivery",
                                            origin_invocation_id=slack_delivery.invocation_id)
                    else:
                        try:
                            check_request_deadline("action_postprocess")
                            invocation_id = uuid4().hex
                            slack_delivery = slack_delivery.bind_text(body)
                            result = invoke_tool("slack_notify", slack_notify_tool, invocation_id=invocation_id,
                                                 text=body, delivery=slack_delivery)
                            slack_delivery = result.model_copy(update={"invocation_id": invocation_id})
                        except LLMCallError as exc:
                            if exc.problem.code != "request_timeout":
                                raise
                            # Keep an earlier save receipt and its pending delivery
                            # state. The runner reports the expired request after
                            # receiving this update; no second side effect starts.
                            record_nonexecution("slack_notify", reason_code="request_timeout")
                    if slack_delivery.status != "pending":
                        messages.append(build_tool_message("slack_notify", slack_delivery.model_dump(mode="json"), 1))
                else:
                    record_nonexecution("slack_notify", reason_code="destination_not_ready")
                if slack_delivery.status != "pending":
                    receipt = build_slack_receipt(slack_delivery)
                    receipts.append(receipt)
                    if receipt.status == "success":
                        completed.add("slack_notify")
                    elif slack_delivery.failure is not None:
                        needs_destination = slack_delivery.failure.next_action == "correct_input"
                        error_codes.append(receipt.error_code)
            else:
                record_nonexecution("slack_notify", reason_code="execution_not_ready")

        if body_ready:
            receipts.extend(ActionReceipt(kind=name, status="skipped", message=_action_question(contract, name))
                            for name in intents if name in waiting_intents)

        for receipt in receipts:
            if receipt.status in {"error", "unknown"}:
                action_errors.append(f"{receipt.kind}: {receipt.error}")

        if receipts:
            updates["response"] = response.model_copy(update={
                "result": response.result.model_copy(update={"actions": [*response.result.actions, *receipts]}),
                "save_operation_binding_sha256": save_operation.binding_sha256 if save_operation is not None else None,
            })
        next_pending = PendingRequest.after_execution(
            contract=contract, previous_response=runtime.previous_response, pending=pending,
            response=response.result, body_ready=body_ready, completed_actions=tuple(sorted(completed)),
            save_operation=save_operation, save_receipt=save_receipt, slack_delivery=slack_delivery,
            receipts=receipts, needs_destination=needs_destination,
        )
        if next_pending is not None or pending is not None:
            updates["runtime"] = runtime.model_copy(update={"pending_request": next_pending})
        if messages:
            updates["messages"] = messages
        if action_errors or error_codes:
            updates["debug"] = debug.model_copy(update={
                "action_errors": [*debug.action_errors, *action_errors],
                "error_codes": list(dict.fromkeys([*debug.error_codes, *error_codes])),
            })
        if verbose and messages:
            log_event(logger, logging.INFO, "postprocess", tools=", ".join(message.name or "" for message in messages))
        if body_ready and save_operation is not None and save_receipt is not None:
            committed_runtime = updates.get("runtime", runtime)
            operations = {**committed_runtime.answer_export_operations,
                          answer_export_key(response.result): save_operation}
            updates["runtime"] = committed_runtime.model_copy(update={"answer_export_operations": operations})
        return updates

    return action_postprocess
