from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage

from ..contracts import GraphState, SlackDestination
from ..contracts.boundary.debug import get_debug_state
from ..contracts.boundary.response import get_response_state
from ..contracts.boundary.runtime import get_runtime_state, parse_session_metadata
from ..logging_utils import log_event
from ..message_utils import build_tool_message
from ..prompts import needs_rag, needs_save, needs_search, needs_slack
from .session import latest_previous_ai_answer


logger = logging.getLogger(__name__)

_CHECKLIST_HINTS = ("checklist", "체크리스트")
_SUMMARY_HINTS = ("summary", "요약")
_PRACTICAL_HINTS = ("실무", "practical")
_BEGINNER_HINTS = ("초보", "beginner", "단계별", "step-by-step")
_ASSUMPTION_HINTS = ("가정", "assumption", "assumptions")
_AMBIGUITY_HINTS = ("모호", "해석 2가지")
_CLARIFICATION_HINTS = ("확인 질문", "clarification")
_CONSTRAINT_HINTS = ("제약", "용어 정의")
_PITFALL_HINTS = ("실수하기 쉬운 사안", "pitfall", "pitfalls")
_LOW_SIGNAL_ACTION_PATTERNS = (
    "현재 대화에는",
    "이전 답변 본문이 없습니다",
    "보낼 내용을 알려",
    "저장할 내용을 알려",
    "공유 요청 조건",
    "전달 결과 텍스트",
    "no previous answer",
    "share the body",
    "share request condition",
    "final message body",
    "internal checklist",
)
_META_ONLY_CHECKLIST_PATTERNS = (
    "체크리스트",
    "공유 요청 조건",
    "메시지 본문",
    "checklist",
    "share request condition",
    "final message body",
)


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
        "검색",
        "찾아",
        "문서",
        "공식",
        "예제",
        "노트북",
        "업로드",
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


def get_slack_destinations(session_metadata: Any) -> SlackDestination:
    metadata = parse_session_metadata(session_metadata)
    return metadata.slack_destination or SlackDestination()


def _normalize_action_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _contains_any(normalized_text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in normalized_text for hint in hints)


def _primary_user_request_line(user_input: str) -> str:
    for raw_line in str(user_input or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("추가 지시:") or line.lower().startswith("additional instruction:"):
            continue
        return line
    return str(user_input or "").strip()


def _requests_summary_checklist(user_input: str) -> bool:
    lowered = _normalize_action_text(user_input)
    return _contains_any(lowered, _CHECKLIST_HINTS) or (
        _contains_any(lowered, _SUMMARY_HINTS) and _contains_any(lowered, _CHECKLIST_HINTS)
    )


def _looks_like_request_echo(*, user_input: str, final_answer: str) -> bool:
    normalized_answer = _normalize_action_text(final_answer)
    normalized_input = _normalize_action_text(user_input)
    return bool(normalized_answer and normalized_answer == normalized_input)


def _looks_like_meta_only_checklist(final_answer: str) -> bool:
    lowered = _normalize_action_text(final_answer)
    return (
        any(marker in lowered for marker in _META_ONLY_CHECKLIST_PATTERNS)
        and any(token in lowered for token in ("checklist", "체크리스트"))
    )


def _needs_action_body_rewrite(*, user_input: str, final_answer: str) -> bool:
    stripped = str(final_answer or "").strip()
    if not stripped:
        return True
    if _looks_like_request_echo(user_input=user_input, final_answer=final_answer):
        return True

    lowered = _normalize_action_text(stripped)
    if any(marker in lowered for marker in _LOW_SIGNAL_ACTION_PATTERNS):
        return True
    return _looks_like_meta_only_checklist(stripped)


def _delivery_label(*, user_input: str) -> str:
    return "공유용 본문" if needs_slack(user_input) else "저장용 본문"


def _infer_delivery_subject(*, user_input: str) -> str:
    request_line = _normalize_action_text(_primary_user_request_line(user_input))
    if "이번 결과" in request_line:
        return "이번 결과"
    if "결과" in request_line:
        return "결과"
    if any(token in request_line for token in ("방금 답변", "이 답변", "답변")):
        return "요청하신 내용"
    return "요청하신 내용"


def _base_delivery_body(*, user_input: str) -> str:
    subject = _infer_delivery_subject(user_input=user_input)
    if needs_save(user_input):
        return (
            f"{subject}를 텍스트로 정리했습니다.\n"
            "- 다시 공유하거나 저장해 재활용하기 쉬운 형태로 적었습니다."
        )
    if needs_slack(user_input):
        return (
            f"{subject}를 전달드립니다.\n"
            "- 바로 공유할 수 있도록 핵심만 간단히 정리했습니다."
        )
    return f"{subject}를 정리했습니다."


def _build_action_delivery_body(*, user_input: str) -> str:
    lowered = _normalize_action_text(user_input)

    sections: list[str] = []
    sections.append(_base_delivery_body(user_input=user_input))
    if _requests_summary_checklist(user_input) or _contains_any(lowered, _SUMMARY_HINTS):
        sections.append(
            "핵심 요약\n"
            "- 핵심만 먼저 보이도록 짧게 정리했습니다."
        )
    if _contains_any(lowered, _ASSUMPTION_HINTS):
        sections.append(
            "가정\n"
            "- 현재 요청에서 확인되는 범위만 반영했습니다."
        )
    if _contains_any(lowered, _CONSTRAINT_HINTS):
        sections.append(
            "제약\n"
            "- 추가 근거가 없는 세부 내용은 임의로 확장하지 않습니다."
        )
    if _contains_any(lowered, _PRACTICAL_HINTS):
        sections.append(
            "실무 관점\n"
            "- 바로 보내거나 저장할 수 있는 본문 형태를 우선하여 간결하게 정리했습니다."
        )
    return "\n\n".join(section for section in sections if section.strip())


def resolve_action_delivery_answer(
    *,
    user_input: str,
    final_answer: str,
    previous_answer: str,
    slack_target_available: bool,
) -> str:
    if needs_slack(user_input) and not slack_target_available:
        return final_answer
    if not is_action_only_request(user_input):
        return final_answer
    if previous_answer.strip():
        return previous_answer.strip()
    if not _needs_action_body_rewrite(user_input=user_input, final_answer=final_answer):
        return final_answer
    return _build_action_delivery_body(user_input=user_input)


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

    return _build_action_delivery_body(user_input=user_input)


def should_short_circuit_action_only(
    *,
    user_input: str,
    messages: list[AnyMessage],
    slack_target_available: bool,
) -> bool:
    if not is_action_only_request(user_input):
        return False
    if needs_slack(user_input) and not slack_target_available:
        return True
    return False


def _tool_status_is_success(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    status = str(result.get("status") or "").strip().lower()
    return status in {"", "ok", "success"}


def _build_save_receipt(save_result: Any) -> str | None:
    if not isinstance(save_result, dict):
        return None
    file_path = str(save_result.get("file_path") or "").strip()
    if _tool_status_is_success(save_result) and file_path:
        return f"저장 완료: {file_path}"
    error = str(save_result.get("error") or save_result.get("message") or "").strip()
    if error:
        return f"저장 실패: {error}"
    return None


def _build_slack_target_label(*, destinations: SlackDestination, slack_result: Any) -> str:
    if isinstance(slack_result, dict):
        for key in ("channel_id", "user_id", "email"):
            value = str(slack_result.get(key) or "").strip()
            if value:
                if key == "channel_id":
                    return f"Slack ({value})"
                return f"Slack DM ({value})"
    if destinations.channel_id:
        return f"Slack ({destinations.channel_id})"
    if destinations.user_id:
        return f"Slack DM ({destinations.user_id})"
    if destinations.email:
        return f"Slack DM ({destinations.email})"
    return "Slack"


def _build_slack_receipt(*, slack_result: Any, destinations: SlackDestination) -> str | None:
    if not isinstance(slack_result, dict):
        return None
    target_label = _build_slack_target_label(destinations=destinations, slack_result=slack_result)
    if _tool_status_is_success(slack_result):
        return f"전송 완료: {target_label}"
    error = str(slack_result.get("error") or slack_result.get("message") or "").strip()
    if error:
        return f"전송 실패: {error}"
    return None


def _compose_action_response_text(*, delivery_body: str, receipts: list[str]) -> str:
    body = str(delivery_body or "").strip()
    receipt_block = "\n".join(receipt for receipt in receipts if receipt.strip()).strip()
    if body and receipt_block:
        return f"{body}\n\n{receipt_block}"
    return body or receipt_block


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
            save_receipt = _build_save_receipt(save_result)
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
            slack_receipt = _build_slack_receipt(slack_result=slack_result, destinations=destinations)
            if slack_receipt:
                receipts.append(slack_receipt)

        final_answer = _compose_action_response_text(delivery_body=delivery_body, receipts=receipts)
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
