from __future__ import annotations

from src.core.planner_schema import PLANNER_ROUTES, PlannerRouteName


RouteName = PlannerRouteName

ROUTE_ORDER = PLANNER_ROUTES
TOOL_TO_ROUTE: dict[str, RouteName] = {
    "tavily_search": "docs",
    "upload_search": "upload",
}


def route_for_tool(tool_name: str) -> str:
    return TOOL_TO_ROUTE.get(str(tool_name or "").strip(), "")
