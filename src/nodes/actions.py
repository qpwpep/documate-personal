from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AnyMessage

from ..logging_utils import log_event
from ..prompts import needs_rag, needs_save, needs_search, needs_slack
from .session import latest_previous_ai_answer
from .state import (
    State,
    build_tool_message,
    coerce_session_metadata,
    coerce_slack_destination,
)


logger = logging.getLogger(__name__)


def has_action_lookup_intent(user_input: str) -> bool:
    lowered = str(user_input or "").lower()
    if not lowered.strip():
        return False

    lookup_keywords = (
        "official",
        "docs",
        "documentation",
        "reference",
        "api",
        "example",
        "sample",
        "notebook",
        ".ipynb",
        ".py",
        "upload",
        "uploaded",
        "file",
        "검색",
        "찾아",
        "문서",
        "공식",
        "예제",
        "노트북",
        "업로드",
        "파일",
        "코드",
        "설명",
    )
    return any(keyword in lowered for keyword in lookup_keywords)


def is_action_only_request(user_input: str) -> bool:
    if not (needs_save(user_input) or needs_slack(user_input)):
        return False
    if needs_search(user_input) or needs_rag(user_input):
        return False
    return not has_action_lookup_intent(user_input)


def get_slack_destinations(session_metadata: Any) -> dict[str, str | None]:
    metadata = coerce_session_metadata(session_metadata)
    return coerce_slack_destination(metadata.get("slack_destination"))


def build_action_only_answer(
    *,
    user_input: str,
    messages: list[AnyMessage],
    slack_target_available: bool,
) -> str:
    if needs_slack(user_input) and not slack_target_available:
        return (
            "Slack으로 공유할 대상을 알려주세요. "
            "channel_id, user_id, 또는 email 중 하나가 필요합니다."
        )

    previous_answer = latest_previous_ai_answer(messages)
    if previous_answer:
        return previous_answer

    if needs_save(user_input) and needs_slack(user_input):
        return "요청하신 최종 답변을 저장하고 Slack으로 공유합니다."
    if needs_save(user_input):
        return "요청하신 최종 답변을 저장합니다."
    if needs_slack(user_input):
        return "요청하신 최종 답변을 Slack으로 공유합니다."
    return ""


def make_action_postprocess_node(
    save_text_tool: Any,
    slack_notify_tool: Any,
    verbose: bool,
    has_default_slack_destination: bool = False,
):
    def action_postprocess(state: State) -> State:
        user_input = str(state.get("user_input", "") or "")
        final_answer = str(state.get("final_answer", "") or "")
        action_errors: list[str] = []
        tool_messages = []

        if (needs_save(user_input) or needs_slack(user_input)) and not final_answer.strip():
            action_errors.append("postprocess: final_answer is empty, skipping save/slack actions")

        if needs_save(user_input) and final_answer.strip():
            try:
                save_result = save_text_tool.func(content=final_answer, filename_prefix="response")
            except Exception as exc:
                save_result = {"status": "error", "error": str(exc)}
                action_errors.append(f"save_text: failed ({exc})")
            tool_messages.append(build_tool_message("save_text", save_result, 1))

        if needs_slack(user_input) and final_answer.strip():
            destinations = get_slack_destinations(state.get("session_metadata"))
            if any(destinations.values()) or has_default_slack_destination:
                try:
                    slack_result = slack_notify_tool.func(
                        text=final_answer,
                        user_id=destinations.get("user_id"),
                        email=destinations.get("email"),
                        channel_id=destinations.get("channel_id"),
                        target="auto",
                    )
                except Exception as exc:
                    slack_result = {"status": "error", "error": str(exc)}
                    action_errors.append(f"slack_notify: failed ({exc})")
                tool_messages.append(build_tool_message("slack_notify", slack_result, 1))

        if verbose and tool_messages:
            tool_names = ", ".join(message.name for message in tool_messages if message.name)
            log_event(logger, logging.INFO, "postprocess", tools=tool_names)

        updates: State = {}
        if tool_messages:
            updates["messages"] = tool_messages
        if action_errors:
            updates["action_errors"] = action_errors
        return updates

    return action_postprocess
