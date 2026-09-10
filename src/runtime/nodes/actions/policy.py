from __future__ import annotations

from typing import Any

from src.core.contracts import SlackDestination
from src.core.contracts.boundary.runtime import parse_session_metadata


def get_slack_destinations(session_metadata: Any) -> SlackDestination:
    metadata = parse_session_metadata(session_metadata)
    return metadata.slack_destination or SlackDestination()
