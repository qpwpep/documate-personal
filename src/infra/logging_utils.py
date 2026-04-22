from __future__ import annotations

import logging
from typing import Any


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s event=%(event)s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class EventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "event"):
            record.event = "log"
        return super().format(record)


def _build_event_formatter() -> EventFormatter:
    return EventFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        current_formatter = getattr(handler, "formatter", None)
        if getattr(handler, "_documate_handler", False) or isinstance(current_formatter, EventFormatter):
            handler.setFormatter(_build_event_formatter())
            root_logger.setLevel(level)
            return

    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler._documate_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(_build_event_formatter())
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
