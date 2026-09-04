from __future__ import annotations

from typing import Iterable

from src.core.planner_schema import PLANNER_ROUTES, PlannerRouteName


RouteName = PlannerRouteName

ROUTE_ORDER = PLANNER_ROUTES
TOOL_TO_ROUTE: dict[str, RouteName] = {
    "tavily_search": "docs",
    "upload_search": "upload",
}


def route_for_tool(tool_name: str) -> str:
    return TOOL_TO_ROUTE.get(str(tool_name or "").strip(), "")


def is_known_route(value: str) -> bool:
    return str(value or "").strip() in ROUTE_ORDER


def normalize_routes(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    normalized = {
        str(value).strip()
        for value in values
        if str(value).strip() in ROUTE_ORDER
    }
    return [route for route in ROUTE_ORDER if route in normalized]


def sort_routes(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    return sorted(
        {str(value).strip() for value in values if str(value).strip()},
        key=lambda value: (ROUTE_ORDER.index(value), value)
        if value in ROUTE_ORDER
        else (len(ROUTE_ORDER), value),
    )
