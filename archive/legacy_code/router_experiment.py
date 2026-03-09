from typing import Literal

from pydantic import BaseModel, Field

RouteIntent = Literal["direct", "search", "rag", "save", "slack"]


class RouteDecision(BaseModel):
    intents: list[RouteIntent] = Field(default_factory=lambda: ["direct"])
    confidence: float = 0.0
    reason: str = ""


def _dedupe_keep_order(items: list[RouteIntent]) -> list[RouteIntent]:
    seen: set[RouteIntent] = set()
    deduped: list[RouteIntent] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def normalize_route_decision(
    decision: RouteDecision,
    has_retriever: bool,
    threshold: float,
) -> tuple[list[RouteIntent], bool, str, float]:
    """Normalize router output into a safe, deterministic routing decision."""
    intents = _dedupe_keep_order(decision.intents or ["direct"])

    if "direct" in intents and len(intents) > 1:
        intents = [intent for intent in intents if intent != "direct"]

    if not has_retriever:
        intents = [intent for intent in intents if intent != "rag"]

    if not intents:
        intents = ["direct"]

    try:
        confidence = float(decision.confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = (decision.reason or "").strip() or "Router did not provide a reason."

    fallback_applied = False
    if confidence < threshold:
        intents = ["direct"]
        fallback_applied = True
        reason = (
            f"{reason} "
            f"(fallback: confidence {confidence:.2f} < threshold {threshold:.2f})"
        )

    return intents, fallback_applied, reason, confidence
