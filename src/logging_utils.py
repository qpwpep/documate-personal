from __future__ import annotations

import logging
from typing import Any


class EventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "event"):
            record.event = "log"
        return super().format(record)


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if any(getattr(handler, "_documate_handler", False) for handler in root_logger.handlers):
        root_logger.setLevel(level)
        return

    if root_logger.handlers:
        for handler in root_logger.handlers:
            current_formatter = getattr(handler, "formatter", None)
            if isinstance(current_formatter, EventFormatter):
                root_logger.setLevel(level)
                return
        return

    handler = logging.StreamHandler()
    handler._documate_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(
        EventFormatter("%(levelname)s %(name)s event=%(event)s %(message)s")
    )
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def _stringify_log_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("\n", "\\n")


def format_log_fields(**fields: Any) -> str:
    parts = [
        f"{key}={_stringify_log_value(value)}"
        for key, value in fields.items()
        if value is not None
    ]
    return " ".join(parts)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    logger.log(level, format_log_fields(**fields), extra={"event": event})
