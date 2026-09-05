from __future__ import annotations

import re

from langchain_core.messages import AnyMessage

from src.core.prompts import needs_save, needs_slack
from src.runtime.nodes.session import latest_previous_ai_answer
from src.runtime.nodes.actions.policy import is_action_only_request


CHECKLIST_HINTS = ("checklist", "체크리스트")
SUMMARY_HINTS = ("summary", "요약")
PRACTICAL_HINTS = ("실무", "practical")
ASSUMPTION_HINTS = ("가정", "assumption", "assumptions")
CONSTRAINT_HINTS = ("제약", "용어 정의")
LOW_SIGNAL_ACTION_PATTERNS = (
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
META_ONLY_CHECKLIST_PATTERNS = (
    "체크리스트",
    "공유 요청 조건",
    "메시지 본문",
    "checklist",
    "share request condition",
    "final message body",
)


def normalize_action_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def contains_any(normalized_text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in normalized_text for hint in hints)


def primary_user_request_line(user_input: str) -> str:
    for raw_line in str(user_input or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("추가 지시:") or line.lower().startswith("additional instruction:"):
            continue
        return line
    return str(user_input or "").strip()


def requests_summary_checklist(user_input: str) -> bool:
    lowered = normalize_action_text(user_input)
    return contains_any(lowered, CHECKLIST_HINTS) or (
        contains_any(lowered, SUMMARY_HINTS) and contains_any(lowered, CHECKLIST_HINTS)
    )


def looks_like_request_echo(*, user_input: str, final_answer: str) -> bool:
    normalized_answer = normalize_action_text(final_answer)
    normalized_input = normalize_action_text(user_input)
    return bool(normalized_answer and normalized_answer == normalized_input)


def looks_like_meta_only_checklist(final_answer: str) -> bool:
    lowered = normalize_action_text(final_answer)
    return (
        any(marker in lowered for marker in META_ONLY_CHECKLIST_PATTERNS)
        and any(token in lowered for token in ("checklist", "체크리스트"))
    )


def needs_action_body_rewrite(*, user_input: str, final_answer: str) -> bool:
    stripped = str(final_answer or "").strip()
    if not stripped:
        return True
    if looks_like_request_echo(user_input=user_input, final_answer=final_answer):
        return True

    lowered = normalize_action_text(stripped)
    if any(marker in lowered for marker in LOW_SIGNAL_ACTION_PATTERNS):
        return True
    return looks_like_meta_only_checklist(stripped)


def infer_delivery_subject(*, user_input: str) -> str:
    request_line = normalize_action_text(primary_user_request_line(user_input))
    if "이번 결과" in request_line:
        return "이번 결과"
    if "결과" in request_line:
        return "결과"
    if any(token in request_line for token in ("방금 답변", "이 답변", "답변")):
        return "요청하신 내용"
    return "요청하신 내용"


def base_delivery_body(*, user_input: str) -> str:
    subject = infer_delivery_subject(user_input=user_input)
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


def build_action_delivery_body(*, user_input: str) -> str:
    lowered = normalize_action_text(user_input)

    sections: list[str] = []
    sections.append(base_delivery_body(user_input=user_input))
    if requests_summary_checklist(user_input) or contains_any(lowered, SUMMARY_HINTS):
        sections.append(
            "핵심 요약\n"
            "- 핵심만 먼저 보이도록 짧게 정리했습니다."
        )
    if contains_any(lowered, ASSUMPTION_HINTS):
        sections.append(
            "가정\n"
            "- 현재 요청에서 확인되는 범위만 반영했습니다."
        )
    if contains_any(lowered, CONSTRAINT_HINTS):
        sections.append(
            "제약\n"
            "- 추가 근거가 없는 세부 내용은 임의로 확장하지 않습니다."
        )
    if contains_any(lowered, PRACTICAL_HINTS):
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
    if not needs_action_body_rewrite(user_input=user_input, final_answer=final_answer):
        return final_answer
    return build_action_delivery_body(user_input=user_input)


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

    return build_action_delivery_body(user_input=user_input)
