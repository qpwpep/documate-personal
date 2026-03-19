from __future__ import annotations

from ...contracts.routes import ROUTE_ORDER
from ...prompts import has_explicit_docs_intent, has_explicit_local_intent, needs_search
from ...rules import get_rules_config


def _planner_rules():
    return get_rules_config().planner


def has_upload_route_intent(user_input: str) -> bool:
    lowered = str(user_input or "").lower()
    return bool(lowered.strip()) and any(keyword in lowered for keyword in _planner_rules().upload_keywords)


def needs_upload_followup(user_input: str) -> bool:
    return has_upload_route_intent(user_input)


def is_upload_only_request(user_input: str) -> bool:
    return has_upload_route_intent(user_input) and not has_explicit_docs_intent(user_input)


def detect_required_routes(user_input: str) -> list[str]:
    trimmed = str(user_input or "").strip()
    if not trimmed:
        return []

    upload_route_intent = has_upload_route_intent(trimmed)
    docs_route_intent = has_explicit_docs_intent(trimmed) if upload_route_intent else needs_search(trimmed)
    local_route_intent = has_explicit_local_intent(trimmed) and not docs_route_intent and not upload_route_intent

    routes: list[str] = []
    if docs_route_intent:
        routes.append("docs")
    if upload_route_intent:
        routes.append("upload")
    elif local_route_intent:
        routes.append("local")
    return [route for route in ROUTE_ORDER if route in routes]
