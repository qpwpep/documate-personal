from __future__ import annotations

from typing import Any

from src.core.contracts import SlackDestination
from src.core.contracts.boundary.runtime import parse_session_metadata
from src.core.prompts import needs_save, needs_search, needs_slack


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
    if needs_search(user_input):
        return False
    return not has_action_lookup_intent(user_input)


def get_slack_destinations(session_metadata: Any) -> SlackDestination:
    metadata = parse_session_metadata(session_metadata)
    return metadata.slack_destination or SlackDestination()


def should_short_circuit_action_only(
    *,
    user_input: str,
    messages: list[Any],
    slack_target_available: bool,
) -> bool:
    _ = messages
    if not is_action_only_request(user_input):
        return False
    if needs_slack(user_input) and not slack_target_available:
        return True
    return False
