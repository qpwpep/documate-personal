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
_LOW_SIGNAL_ACTION_PATTERNS = (
    "알려주시면",
    "보내주시면",
    "제공되지 않았습니다",
    "생성할 수 없습니다",
    "저장할 내용을",
    "전달할 결과 텍스트",
    "현재 대화에는",
    "무엇을 답할지",
    "무엇의 최종 답변",
)
_META_ONLY_CHECKLIST_PATTERNS = (
    "저장/공유 요청 조건",
    "슬랙 전송용",
    "메시지 본문",
    "바로 붙여넣을 수 있는",
    "최종 답변",
)
_PRACTICAL_HINTS = ("실무", "practical")
_BEGINNER_HINTS = ("초보자", "beginner", "단계별", "step-by-step")
_ASSUMPTION_HINTS = ("전제", "assumption", "입력 정보가 부족", "assumptions")
_AMBIGUITY_HINTS = ("가능한 해석 2가지", "해석 2가지", "모호한 부분")
_CLARIFICATION_HINTS = ("확인 질문", "범위를 좁히는 확인 질문", "clarification")
_CONSTRAINT_HINTS = ("가정/제약", "가정", "제약", "용어 정의가 모호")
_PITFALL_HINTS = ("실수하기 쉬운 포인트", "pitfall", "pitfalls")


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


def _requests_summary_checklist(user_input: str) -> bool:
    lowered = _normalize_action_text(user_input)
    return any(hint in lowered for hint in _CHECKLIST_HINTS) or (
        any(hint in lowered for hint in _SUMMARY_HINTS)
        and any(hint in lowered for hint in _CHECKLIST_HINTS)
    )


def _looks_like_request_echo(*, user_input: str, final_answer: str) -> bool:
    normalized_answer = _normalize_action_text(final_answer)
    normalized_input = _normalize_action_text(user_input)
    return bool(normalized_answer and normalized_answer == normalized_input)


def _looks_like_meta_only_checklist(final_answer: str) -> bool:
    lowered = _normalize_action_text(final_answer)
    return "체크리스트" in lowered and any(
        marker in lowered for marker in _META_ONLY_CHECKLIST_PATTERNS
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


def _build_action_delivery_body(*, user_input: str) -> str:
    lowered = _normalize_action_text(user_input)
    is_slack = needs_slack(user_input)
    is_save = needs_save(user_input)
    delivery_label = "공유 내용" if is_slack else "저장 내용"
    delivery_sentence = (
        "이 메시지를 그대로 Slack DM으로 전달합니다."
        if is_slack and "dm" in lowered
        else (
            "이 메시지를 그대로 Slack으로 전달합니다."
            if is_slack
            else (
                "이 메시지를 그대로 텍스트 파일에 저장합니다."
                if is_save
                else "이 메시지를 그대로 저장하거나 전송합니다."
            )
        )
    )

    sections: list[str] = []

    if any(hint in lowered for hint in _ASSUMPTION_HINTS):
        sections.append(
            "전제\n"
            "- 현재 대화에는 재사용할 이전 답변 본문이 없습니다.\n"
            f"- {delivery_sentence}"
        )

    if any(hint in lowered for hint in _AMBIGUITY_HINTS):
        sections.append(
            "가능한 해석 1\n"
            "- 직전 답변 본문을 그대로 저장하거나 공유해 달라는 요청입니다.\n\n"
            "가능한 해석 2\n"
            "- 직전 답변 본문이 없으면 현재 턴에서 대체 본문을 만들어 처리해 달라는 요청입니다.\n\n"
            "선택한 처리\n"
            "- 현재 대화에는 재사용할 직전 본문이 없어 2번 해석을 기준으로 처리합니다."
        )

    if any(hint in lowered for hint in _CLARIFICATION_HINTS):
        sections.append(
            "확인 질문\n"
            "- 저장 또는 공유 대상이 직전 답변 전체인지, 핵심 요약인지 확인이 필요합니다.\n\n"
            "현재 처리\n"
            "- 직전 답변 본문이 없어 우선 이 메시지를 임시 최종 본문으로 사용합니다."
        )

    if any(hint in lowered for hint in _CONSTRAINT_HINTS):
        sections.append(
            "가정\n"
            "- 현재 대화에는 재사용할 이전 답변 본문이 없습니다.\n\n"
            "제약\n"
            "- 존재하지 않는 이전 답변 내용을 임의로 만들어 넣지 않습니다."
        )

    if _requests_summary_checklist(user_input):
        sections.append(
            "요약\n"
            "- 현재 대화에는 재사용할 이전 답변 본문이 없습니다.\n"
            "- 따라서 추가 질문 대신 바로 저장하거나 전송할 수 있는 최소 본문으로 정리합니다.\n\n"
            "체크리스트\n"
            "- [x] 추가 질문 없이 바로 저장/전송 가능\n"
            "- [x] 현재 턴 기준의 최종 본문으로 확정됨\n"
            "- [x] 근거 없는 새 내용은 생성하지 않음"
        )

    if any(hint in lowered for hint in _PRACTICAL_HINTS):
        sections.append(
            "핵심 요약\n"
            "- 현재 대화에는 재사용할 이전 답변 본문이 없습니다.\n"
            "- 이번 요청은 추가 확인보다 즉시 사용 가능한 전달본 확보가 우선입니다.\n\n"
            "실무 관점\n"
            "- 자동 저장 또는 공유 요청에서는 존재하지 않는 이전 본문을 추정하지 않고, 즉시 사용할 수 있는 대체 본문으로 처리하는 편이 안전합니다."
        )

    if any(hint in lowered for hint in _BEGINNER_HINTS):
        sections.append(
            "단계별 안내\n"
            "1. 재사용할 이전 답변 본문이 있는지 먼저 확인합니다.\n"
            "2. 현재 대화에서는 이전 본문이 없어 추가 질문 대신 전달 가능한 최소 본문을 확정합니다.\n"
            f"3. 확정한 본문을 기준으로 {delivery_sentence}"
        )

    if any(hint in lowered for hint in _PITFALL_HINTS):
        sections.append(
            "실수하기 쉬운 포인트\n"
            "- 존재하지 않는 이전 답변을 있는 것처럼 가정하지 않습니다.\n"
            "- 수신 대상 정보와 실제 전달 본문을 혼동하지 않습니다."
        )

    sections.append(
        f"{delivery_label}\n"
        "- 현재 대화에는 재사용할 이전 답변 본문이 없습니다.\n"
        "- 따라서 이 메시지 자체를 이번 요청의 최종 전달본으로 사용합니다.\n"
        f"- {delivery_sentence}"
    )

    return "\n\n".join(section for section in sections if section.strip())


def resolve_action_delivery_answer(
    *,
    user_input: str,
    final_answer: str,
    slack_target_available: bool,
) -> str:
    if needs_slack(user_input) and not slack_target_available:
        return final_answer
    if not is_action_only_request(user_input):
        return final_answer
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
    _ = messages
    return True


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
        destinations = get_slack_destinations(runtime.session_metadata)
        slack_target_available = destinations.has_destination() or has_default_slack_destination
        final_answer = resolve_action_delivery_answer(
            user_input=user_input,
            final_answer=response.final_answer,
            slack_target_available=slack_target_available,
        )
        action_errors: list[str] = []
        tool_messages = []
        response_override_messages = []

        if final_answer.strip() and final_answer.strip() != str(response.final_answer or "").strip():
            response_override_messages.append(AIMessage(content=final_answer))

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
            if slack_target_available:
                try:
                    slack_result = slack_notify_tool.func(
                        text=final_answer,
                        user_id=destinations.user_id,
                        email=destinations.email,
                        channel_id=destinations.channel_id,
                        target="auto",
                    )
                except Exception as exc:
                    slack_result = {"status": "error", "error": str(exc)}
                    action_errors.append(f"slack_notify: failed ({exc})")
                tool_messages.append(build_tool_message("slack_notify", slack_result, 1))

        if verbose and tool_messages:
            tool_names = ", ".join(message.name for message in tool_messages if message.name)
            log_event(logger, logging.INFO, "postprocess", tools=tool_names)

        updates: GraphState = {}
        if response_override_messages or tool_messages:
            updates["messages"] = [*response_override_messages, *tool_messages]
        if action_errors:
            updates["debug"] = debug.model_copy(
                update={"action_errors": [*debug.action_errors, *action_errors]}
            )
        return updates

    return action_postprocess
