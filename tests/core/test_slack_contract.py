from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.slack_contract import ExplicitRecipient, SlackDelivery


def _serialized_user_delivery(**updates):
    """A serialized pending result as received from an internal caller/storage."""
    return {
        "intent": {"state": "explicit", "selector": {"kind": "user", "value": "UINTENDED"}},
        "selection": {"request_id": "request", "source": "request_input", "selector": {"kind": "user", "value": "UINTENDED"}},
        "resolved_user_id": "UINTENDED",
        "target": {"channel_id": "DINTENDED", "user_id": "UINTENDED"},
        "status": "pending",
        **updates,
    }










