from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from src.core.contracts import GraphState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.infra.logging_utils import log_event
from src.core.message_utils import build_tool_message
from src.core.prompts import needs_save, needs_slack
from src.runtime.nodes.session import latest_previous_ai_answer
from src.runtime.nodes.actions.delivery import resolve_action_delivery_answer
from src.runtime.nodes.actions.policy import get_slack_destinations
from src.runtime.nodes.actions.receipts import build_save_receipt, build_slack_receipt, compose_action_response_text


logger = logging.getLogger(__name__)


def make_action_postprocess_node(
    save_text_tool: Any,
    slack_notify_tool: Any,
    verbose: bool,
    has_default_slack_destination: bool = False,
):
    def action_postprocess(state: GraphState) -> GraphState:
        runtime = get_runtime_state(state)
        response = get_response_state(state)
        debug = get_debug_state(state)
        user_input = runtime.user_input
        previous_answer = latest_previous_ai_answer(state.get("messages", []))
        destinations = get_slack_destinations(runtime.session_metadata)
        slack_target_available = destinations.has_destination() or has_default_slack_destination
        delivery_body = resolve_action_delivery_answer(
            user_input=user_input,
            final_answer=response.final_answer,
            previous_answer=previous_answer,
            slack_target_available=slack_target_available,
        )

        action_errors: list[str] = []
        tool_messages = []
        response_override_messages = []
        receipts: list[str] = []

        if (needs_save(user_input) or needs_slack(user_input)) and not delivery_body.strip():
            action_errors.append("postprocess: delivery_body is empty, skipping save/slack actions")

        if needs_save(user_input) and delivery_body.strip():
            try:
                save_result = save_text_tool.func(content=delivery_body, filename_prefix="response")
            except Exception as exc:
                save_result = {"status": "error", "error": str(exc)}
                action_errors.append(f"save_text: failed ({exc})")
            tool_messages.append(build_tool_message("save_text", save_result, 1))
            save_receipt = build_save_receipt(save_result)
            if save_receipt:
                receipts.append(save_receipt)

        if needs_slack(user_input) and delivery_body.strip() and slack_target_available:
            try:
                slack_result = slack_notify_tool.func(
                    text=delivery_body,
                    user_id=destinations.user_id,
                    email=destinations.email,
                    channel_id=destinations.channel_id,
                    target="auto",
                )
            except Exception as exc:
                slack_result = {"status": "error", "error": str(exc)}
                action_errors.append(f"slack_notify: failed ({exc})")
            tool_messages.append(build_tool_message("slack_notify", slack_result, 1))
            slack_receipt = build_slack_receipt(slack_result=slack_result, destinations=destinations)
            if slack_receipt:
                receipts.append(slack_receipt)

        final_answer = compose_action_response_text(delivery_body=delivery_body, receipts=receipts)
        response_changed = bool(
            final_answer.strip() and final_answer.strip() != str(response.final_answer or "").strip()
        )
        if final_answer.strip() and (response_changed or needs_save(user_input) or needs_slack(user_input)):
            response_override_messages.append(AIMessage(content=final_answer))

        if verbose and tool_messages:
            tool_names = ", ".join(message.name for message in tool_messages if message.name)
            log_event(logger, logging.INFO, "postprocess", tools=tool_names)

        updates: GraphState = {}
        if response_changed:
            updates["response"] = response.model_copy(
                update={
                    "final_answer": final_answer,
                    "payload": response.payload.model_copy(update={"answer": final_answer}),
                    "synthesis_output": response.synthesis_output.model_copy(update={"answer": final_answer}),
                }
            )
        if response_override_messages or tool_messages:
            updates["messages"] = [*response_override_messages, *tool_messages]
        if action_errors:
            updates["debug"] = debug.model_copy(
                update={"action_errors": [*debug.action_errors, *action_errors]}
            )
        return updates

    return action_postprocess
