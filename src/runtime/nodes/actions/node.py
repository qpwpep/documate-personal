from __future__ import annotations

import logging
from typing import Any

from src.core.answer_schema import ActionReceipt, export_answer_text
from src.core.contracts import GraphState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.message_utils import build_tool_message
from src.core.prompts import needs_save, needs_slack
from src.infra.logging_utils import log_event
from src.runtime.nodes.actions.policy import get_slack_destinations
from src.runtime.nodes.actions.receipts import build_save_receipt, build_slack_receipt


logger = logging.getLogger(__name__)


def make_action_postprocess_node(
    save_text_tool: Any, slack_notify_tool: Any, verbose: bool,
    has_default_slack_destination: bool = False,
):
    def action_postprocess(state: GraphState) -> GraphState:
        planner = get_planner_state(state)
        if str(planner.guided_followup or "").strip():
            return {}
        runtime = get_runtime_state(state)
        response = get_response_state(state)
        debug = get_debug_state(state)
        save_requested, slack_requested = needs_save(runtime.user_input), needs_slack(runtime.user_input)
        if not save_requested and not slack_requested:
            return {}

        # Synthesis selects the current or previous AnswerResponse before this stage.
        # Delivery and the UI therefore share one finalized document and its citations.
        body = export_answer_text(response.result, include_sources=True)
        destinations = get_slack_destinations(runtime.session_metadata)
        slack_available = destinations.has_destination() or has_default_slack_destination
        receipts: list[ActionReceipt] = []
        messages = []
        action_errors: list[str] = []
        error_codes = []

        if save_requested:
            if not body.strip():
                receipts.append(ActionReceipt(kind="save_text", status="skipped", message="저장할 본문이 없습니다."))
            else:
                try:
                    result = save_text_tool(content=body, filename_prefix="response")
                except Exception as exc:
                    result = {"status": "error", "error": str(exc)}
                messages.append(build_tool_message("save_text", result, 1))
                receipts.append(build_save_receipt(result))

        if slack_requested:
            if not body.strip():
                receipts.append(ActionReceipt(kind="slack_notify", status="skipped", message="전송할 본문이 없습니다."))
            elif not slack_available:
                receipts.append(ActionReceipt(kind="slack_notify", status="skipped", message="Slack 전송 대상이 필요합니다."))
            else:
                try:
                    result = slack_notify_tool(
                        text=body, user_id=destinations.user_id,
                        email=destinations.email, channel_id=destinations.channel_id,
                    )
                except Exception as exc:
                    result = {"status": "error", "error": str(exc), "error_code": "SLACK_AUTH_FAILED"}
                messages.append(build_tool_message("slack_notify", result, 1))
                receipts.append(build_slack_receipt(slack_result=result, destinations=destinations))
                if isinstance(result, dict) and result.get("error_code"):
                    error_codes.append(result["error_code"])

        for receipt in receipts:
            if receipt.status == "error":
                action_errors.append(f"{receipt.kind}: {receipt.error}")

        updates: GraphState = {
            "response": response.model_copy(update={
                "result": response.result.model_copy(update={"actions": [*response.result.actions, *receipts]}),
            }),
        }
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
